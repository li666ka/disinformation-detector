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
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from api.auth import get_current_user
from api.database import Ensemble, ModelRecord, User, get_db
from api.ensemble_voting import evaluate_ensemble
from api.schemas import EnsembleCreate, EnsembleResponse, EnsembleSummary

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ensembles", tags=["ensembles"])


def _has_predictions(model: ModelRecord) -> bool:
    return bool(model.predictions_json and model.predictions_json.strip())


def _load_member(model: ModelRecord) -> dict:
    """Завантажити predictions для члена ансамблю — з БД."""
    if not _has_predictions(model):
        raise HTTPException(
            status_code=404,
            detail=(
                f"Predictions not found for model {model.id} ({model.name}). "
                "Перетренуйте модель — predictions автоматично запишуться у БД."
            ),
        )
    try:
        data = json.loads(model.predictions_json)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Corrupted predictions JSON for {model.name}: {e}",
        )

    return {
        "article_ids": list(data.get("article_ids", [])),
        "y_true": np.array(data.get("y_true", []), dtype=np.int64),
        "y_pred": np.array(data.get("y_pred", []), dtype=np.int64),
        "y_proba_fake": np.array(data.get("y_proba_fake", []), dtype=np.float32),
        "metrics": {},
        "splits_used": data.get("splits_used", model.splits_used or ""),
        "dataset_id": str(data.get("dataset_id", model.dataset_id or "")),
        "model_type": data.get("model_type", model.model_type),
        "test_size": data.get("test_size", len(data.get("article_ids", []))),
        "model_record_id": model.id,
        "model_name": model.name,
        "model_type_db": model.model_type,
    }


def _model_extra_metrics(record: ModelRecord) -> tuple[Optional[float], Optional[float]]:
    """Дістати f1_macro / roc_auc з metrics_json (не SQL колонки)."""
    if not record.metrics_json:
        return None, None
    try:
        m = json.loads(record.metrics_json)
        return m.get("f1_macro"), m.get("roc_auc")
    except Exception:
        return None, None


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
            "has_predictions": _has_predictions(m),
            "predictions_path": None,
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

    datasets = {m.dataset_id for m in models if m.dataset_id}
    if len(datasets) > 1:
        logger.warning(f"Mixed dataset_ids in ensemble: {datasets}")
    dataset_id = next(iter(datasets), None)

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

    n_aligned = len(result["article_ids"])
    member_test_sizes: list[int] = []
    for m in models:
        size = n_aligned
        if m.metrics_json:
            try:
                mm = json.loads(m.metrics_json)
                size = int(mm.get("test_size") or n_aligned)
            except Exception:
                size = n_aligned
        member_test_sizes.append(size)

    alignment_info = {
        "common_test_size": n_aligned,
        "member_test_sizes": member_test_sizes,
        "max_member_size": max(member_test_sizes) if member_test_sizes else n_aligned,
    }
    logger.info(
        f"Ensemble '{req.name}' aligned: common={n_aligned}, "
        f"members={member_test_sizes}"
    )

    metrics_with_alignment = {**metrics, "alignment_info": alignment_info}

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
        metrics_json=json.dumps(metrics_with_alignment),
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

    n_aligned = len(result["article_ids"])
    member_test_sizes: list[int] = []
    for m in models:
        size = n_aligned
        if m.metrics_json:
            try:
                mm = json.loads(m.metrics_json)
                size = int(mm.get("test_size") or n_aligned)
            except Exception:
                size = n_aligned
        member_test_sizes.append(size)

    alignment_info = {
        "common_test_size": n_aligned,
        "member_test_sizes": member_test_sizes,
        "max_member_size": max(member_test_sizes) if member_test_sizes else n_aligned,
    }
    metrics_with_alignment = {**metrics, "alignment_info": alignment_info}

    ensemble.accuracy = metrics.get("accuracy")
    ensemble.precision = metrics.get("precision")
    ensemble.recall = metrics.get("recall")
    ensemble.f1_score = metrics.get("f1_score")
    ensemble.f1_macro = metrics.get("f1_macro")
    ensemble.roc_auc = metrics.get("roc_auc")
    ensemble.metrics_json = json.dumps(metrics_with_alignment)
    db.commit()
    db.refresh(ensemble)

    return _serialize_ensemble(ensemble, models)


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
    alignment_info = None
    if ensemble.metrics_json:
        try:
            full = json.loads(ensemble.metrics_json)
            cm = full.get("confusion_matrix")
            alignment_info = full.get("alignment_info")
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
        alignment_info=alignment_info,
        splits_used=ensemble.splits_used,
        dataset_id=ensemble.dataset_id,
        is_active=ensemble.is_active,
        created_at=ensemble.created_at,
    )
