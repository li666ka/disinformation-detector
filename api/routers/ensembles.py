"""API endpoints для ансамблів моделей.

CRUD:
  POST   /ensembles                   — створити + evaluate
  GET    /ensembles                   — список (compact)
  GET    /ensembles/{id}              — деталі
  DELETE /ensembles/{id}              — видалити
  POST   /ensembles/{id}/activate     — активувати (інші стають неактивні)
  POST   /ensembles/{id}/evaluate     — re-evaluate
  GET    /ensembles/eligible-models   — моделі готові для ансамблю
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from api.auth import get_current_user
from api.database import Dataset, Ensemble, ModelRecord, User, get_db
from api.ensemble_voting import evaluate_ensemble, load_predictions
from api.schemas import EnsembleCreate, EnsembleResponse, EnsembleSummary

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ensembles", tags=["ensembles"])


# ── Helpers: знайти predictions.json для моделі ────────────────────────

def _models_root() -> Path:
    return Path(os.environ.get("MODELS_ROOT", "models"))


def _candidate_dirs_for_model(model: ModelRecord) -> list[Path]:
    """Можливі директорії де може лежати predictions.json для цієї моделі."""
    out: list[Path] = []
    if not model.model_path:
        return out

    p = Path(model.model_path)
    parent = p.parent if p.suffix == ".pkl" else p
    out.append(parent)  # NB: user_X/nb_<exp>/

    # DistilBERT/GNN: model_path = user_X/model_<exp>.pkl; subdir = user_X/<type>_<exp>/
    if p.suffix == ".pkl" and p.stem.startswith("model_"):
        exp_id = p.stem[len("model_"):]
        type_prefix_map = {
            "nb": "nb",
            "distilbert": "distilbert",
            "deberta": "deberta",
            "bert": "distilbert",
            "gin": "gnn",
            "sage": "gnn",
            "gnn": "gnn",
        }
        prefix = type_prefix_map.get((model.model_type or "").lower())
        if prefix:
            out.append(p.parent / f"{prefix}_{exp_id}")
            # Деякі NB legacy: model_nb_<exp>.pkl → exp_id вже містить "nb_"
            if exp_id.startswith("nb_"):
                out.append(p.parent / exp_id)

    # Якщо bundle вказує на model_dir — peek
    try:
        import joblib  # lazy: тяжкий
        if p.exists() and p.is_file():
            bundle = joblib.load(p)
            if isinstance(bundle, dict) and "model_dir" in bundle:
                out.append(Path(bundle["model_dir"]))
    except Exception:
        pass

    return out


def _model_predictions_path(model: ModelRecord) -> Optional[Path]:
    for d in _candidate_dirs_for_model(model):
        for fname in ("predictions.json", f"predictions_{model.id}.json"):
            cand = d / fname
            if cand.exists():
                return cand
    return None


def _llm_predictions_path(model: ModelRecord) -> Optional[Path]:
    """Для LLM presets predictions лежать у llm_predictions/preset_{id}_*/"""
    if (model.model_type or "").lower() != "llm":
        return None
    root = _models_root() / "llm_predictions"
    if not root.exists():
        return None
    for d in root.glob(f"preset_{model.id}_*"):
        cand = d / "predictions.json"
        if cand.exists():
            return cand
    return None


def _resolve_predictions(model: ModelRecord) -> Optional[Path]:
    return _model_predictions_path(model) or _llm_predictions_path(model)


def _load_member(model: ModelRecord) -> dict:
    preds_path = _resolve_predictions(model)
    if not preds_path:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Predictions not found for model {model.id} ({model.name}). "
                "Re-train модель після впровадження predictions cache."
            ),
        )
    try:
        data = load_predictions(preds_path)
    except Exception as e:
        logger.error(f"Failed to load predictions for model {model.id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load predictions for {model.name}: {e}",
        )
    data["model_record_id"] = model.id
    data["model_name"] = model.name
    data["model_type_db"] = model.model_type
    return data


def _model_extra_metrics(record: ModelRecord) -> tuple[Optional[float], Optional[float]]:
    """Дістати f1_macro / roc_auc з metrics_json (не SQL колонки)."""
    if not record.metrics_json:
        return None, None
    try:
        m = json.loads(record.metrics_json)
        return m.get("f1_macro"), m.get("roc_auc")
    except Exception:
        return None, None


# ── Endpoints ───────────────────────────────────────────────────────────

@router.get("/eligible-models")
def list_eligible_models(
    splits_used: Optional[str] = Query(None),
    dataset_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """Моделі готові для додавання в ансамбль (мають predictions.json)."""
    q = db.query(ModelRecord)
    if splits_used:
        q = q.filter(ModelRecord.splits_used == splits_used)
    if dataset_id:
        q = q.filter(ModelRecord.dataset_id == dataset_id)
    models = q.order_by(ModelRecord.created_at.desc()).all()

    eligible = []
    for m in models:
        preds_path = _resolve_predictions(m)
        f1_macro, roc_auc = _model_extra_metrics(m)
        eligible.append({
            "id": m.id,
            "name": m.name,
            "model_type": m.model_type,
            "splits_used": m.splits_used,
            "dataset_id": m.dataset_id,
            "f1_score": m.f1_score,
            "f1_macro": f1_macro,
            "roc_auc": roc_auc,
            "accuracy": m.accuracy,
            "has_predictions": preds_path is not None,
            "predictions_path": str(preds_path) if preds_path else None,
        })

    return {
        "models": eligible,
        "total": len(eligible),
        "with_predictions": sum(1 for e in eligible if e["has_predictions"]),
    }


@router.post("", response_model=EnsembleResponse)
def create_ensemble(
    req: EnsembleCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Створити новий ансамбль + одразу evaluate."""
    models = (
        db.query(ModelRecord)
        .filter(ModelRecord.id.in_(req.member_model_ids))
        .all()
    )
    found_ids = {m.id for m in models}
    missing = [mid for mid in req.member_model_ids if mid not in found_ids]
    if missing:
        raise HTTPException(status_code=404, detail=f"Models not found: {missing}")

    # ── Validation: same splits_used ──
    splits = {m.splits_used for m in models if m.splits_used}
    if len(splits) > 1:
        raise HTTPException(
            status_code=400,
            detail=(
                f"All members must have the same splits_used. Found: {splits}. "
                "Cannot mix in-domain and cross-domain models."
            ),
        )
    splits_used = splits.pop() if splits else None

    # ── Soft check: dataset ──
    datasets = {m.dataset_id for m in models if m.dataset_id}
    if len(datasets) > 1:
        logger.warning(f"Mixed dataset_ids in ensemble: {datasets}")
    dataset_id = next(iter(datasets), None)

    # ── Load predictions у тому ж порядку, що member_model_ids ──
    by_id = {m.id: m for m in models}
    members_data = [_load_member(by_id[mid]) for mid in req.member_model_ids]

    splits_at_preds = {m.get("splits_used", "") for m in members_data if m.get("splits_used")}
    if len(splits_at_preds) > 1:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Predictions splits mismatch: {splits_at_preds}. "
                "Re-train або re-evaluate моделі."
            ),
        )

    # ── Voting ──
    try:
        result = evaluate_ensemble(
            voting_type=req.voting_type,
            members_data=members_data,
            weights=req.weights,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Ensemble evaluation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Voting failed: {e}")

    metrics = result["metrics"]

    ensemble = Ensemble(
        user_id=current_user.id,
        name=req.name,
        voting_type=req.voting_type,
        member_model_ids=json.dumps(req.member_model_ids),
        weights=json.dumps(req.weights) if req.weights else None,
        accuracy=metrics.get("accuracy"),
        precision=metrics.get("precision"),
        recall=metrics.get("recall"),
        f1_score=metrics.get("f1_score"),
        f1_macro=metrics.get("f1_macro"),
        roc_auc=metrics.get("roc_auc"),
        metrics_json=json.dumps(metrics),
        splits_used=splits_used,
        dataset_id=dataset_id,
        is_active=False,
    )
    db.add(ensemble)
    db.commit()
    db.refresh(ensemble)

    f1m = metrics.get("f1_macro")
    logger.info(
        f"Created ensemble id={ensemble.id} '{req.name}' "
        f"({req.voting_type}, {len(models)} members): "
        f"F1-macro={f1m:.4f}" if f1m is not None else
        f"Created ensemble id={ensemble.id} '{req.name}' "
        f"({req.voting_type}, {len(models)} members)"
    )

    return _serialize_ensemble(ensemble, models)


@router.get("", response_model=list[EnsembleSummary])
def list_ensembles(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Compact список ансамблів користувача."""
    ensembles = (
        db.query(Ensemble)
        .filter(Ensemble.user_id == current_user.id)
        .order_by(Ensemble.created_at.desc())
        .all()
    )

    out: list[EnsembleSummary] = []
    for e in ensembles:
        member_ids = json.loads(e.member_model_ids or "[]")
        out.append(EnsembleSummary(
            id=e.id,
            name=e.name,
            voting_type=e.voting_type,
            member_count=len(member_ids),
            accuracy=e.accuracy,
            f1_macro=e.f1_macro,
            splits_used=e.splits_used,
            is_active=e.is_active,
            created_at=e.created_at,
        ))
    return out


@router.get("/{ensemble_id}", response_model=EnsembleResponse)
def get_ensemble(
    ensemble_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    ensemble = (
        db.query(Ensemble)
        .filter(Ensemble.id == ensemble_id, Ensemble.user_id == current_user.id)
        .first()
    )
    if not ensemble:
        raise HTTPException(status_code=404, detail="Ensemble not found")

    member_ids = json.loads(ensemble.member_model_ids or "[]")
    models = db.query(ModelRecord).filter(ModelRecord.id.in_(member_ids)).all()
    return _serialize_ensemble(ensemble, models)


@router.delete("/{ensemble_id}")
def delete_ensemble(
    ensemble_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    ensemble = (
        db.query(Ensemble)
        .filter(Ensemble.id == ensemble_id, Ensemble.user_id == current_user.id)
        .first()
    )
    if not ensemble:
        raise HTTPException(status_code=404, detail="Ensemble not found")

    db.delete(ensemble)
    db.commit()
    logger.info(f"Deleted ensemble id={ensemble_id}")
    return {"ok": True, "deleted_id": ensemble_id}


@router.post("/{ensemble_id}/activate")
def activate_ensemble(
    ensemble_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    ensemble = (
        db.query(Ensemble)
        .filter(Ensemble.id == ensemble_id, Ensemble.user_id == current_user.id)
        .first()
    )
    if not ensemble:
        raise HTTPException(status_code=404, detail="Ensemble not found")

    db.query(Ensemble).filter(
        Ensemble.user_id == current_user.id,
        Ensemble.id != ensemble_id,
    ).update({"is_active": False})

    ensemble.is_active = True
    db.commit()
    db.refresh(ensemble)
    return {"ok": True, "activated_id": ensemble_id}


@router.post("/{ensemble_id}/evaluate", response_model=EnsembleResponse)
def re_evaluate_ensemble(
    ensemble_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    ensemble = (
        db.query(Ensemble)
        .filter(Ensemble.id == ensemble_id, Ensemble.user_id == current_user.id)
        .first()
    )
    if not ensemble:
        raise HTTPException(status_code=404, detail="Ensemble not found")

    member_ids = json.loads(ensemble.member_model_ids or "[]")
    weights = json.loads(ensemble.weights) if ensemble.weights else None

    models = db.query(ModelRecord).filter(ModelRecord.id.in_(member_ids)).all()
    if len(models) != len(member_ids):
        raise HTTPException(
            status_code=400,
            detail="Some member models were deleted.",
        )

    by_id = {m.id: m for m in models}
    members_data = [_load_member(by_id[mid]) for mid in member_ids]

    try:
        result = evaluate_ensemble(
            voting_type=ensemble.voting_type,
            members_data=members_data,
            weights=weights,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    metrics = result["metrics"]
    ensemble.accuracy = metrics.get("accuracy")
    ensemble.precision = metrics.get("precision")
    ensemble.recall = metrics.get("recall")
    ensemble.f1_score = metrics.get("f1_score")
    ensemble.f1_macro = metrics.get("f1_macro")
    ensemble.roc_auc = metrics.get("roc_auc")
    ensemble.metrics_json = json.dumps(metrics)
    db.commit()
    db.refresh(ensemble)

    return _serialize_ensemble(ensemble, models)


# ── Serializer ──────────────────────────────────────────────────────────

def _serialize_ensemble(
    ensemble: Ensemble,
    models: list[ModelRecord],
) -> EnsembleResponse:
    member_ids = json.loads(ensemble.member_model_ids or "[]")
    weights = json.loads(ensemble.weights) if ensemble.weights else None

    members_info: list[dict] = []
    for mid in member_ids:
        m = next((mm for mm in models if mm.id == mid), None)
        if not m:
            continue
        f1_macro, _ = _model_extra_metrics(m)
        members_info.append({
            "id": m.id,
            "name": m.name,
            "model_type": m.model_type,
            "f1_score": m.f1_score,
            "f1_macro": f1_macro,
            "accuracy": m.accuracy,
            "weight": weights.get(str(mid)) if weights else None,
        })

    cm = None
    if ensemble.metrics_json:
        try:
            full = json.loads(ensemble.metrics_json)
            cm = full.get("confusion_matrix")
        except Exception:
            pass

    return EnsembleResponse(
        id=ensemble.id,
        name=ensemble.name,
        voting_type=ensemble.voting_type,
        member_model_ids=member_ids,
        weights=weights,
        member_models=members_info,
        accuracy=ensemble.accuracy,
        precision=ensemble.precision,
        recall=ensemble.recall,
        f1_score=ensemble.f1_score,
        f1_macro=ensemble.f1_macro,
        roc_auc=ensemble.roc_auc,
        confusion_matrix=cm,
        splits_used=ensemble.splits_used,
        dataset_id=ensemble.dataset_id,
        is_active=ensemble.is_active,
        created_at=ensemble.created_at,
    )
