import os
import json
import logging
import requests
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel, Field
from api.database import SessionLocal, get_db, ModelRecord, Dataset, User
from api.auth import get_current_user
from api.schemas import ModelRecordResponse
from api.llm_evaluator import evaluate_llm_preset
from api.llm_jobs import (
    create_job,
    get_status,
    mark_done,
    mark_failed,
    update_progress,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["models"])


class EvaluateModelRequest(BaseModel):
    max_samples: int = Field(default=100, ge=10, le=2000)
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    splits_subdir: Optional[str] = Field(
        default="splits_cross_domain",
        description="splits_in_domain | splits_cross_domain | splits_mixed",
    )


_EXTRA_METRIC_KEYS = ("f1_macro", "roc_auc")


def _serialize_model(
    record: ModelRecord,
    dataset_name: str | None = None,
) -> ModelRecordResponse:
    """ModelRecord → ModelRecordResponse, з полями розпарсеними із metrics_json.

    Старі записи зберігали лише `accuracy` у SQL колонках, а решту метрик —
    тільки у `metrics_json`. Тому коли SQL колонка NULL, fallback'имось у JSON
    (включаючи варіанти ключів `*_macro` від ML server).
    """
    resp = ModelRecordResponse.model_validate(record)
    if dataset_name is not None:
        resp.dataset_name = dataset_name
    if not record.metrics_json:
        return resp

    try:
        parsed = json.loads(record.metrics_json)
    except (json.JSONDecodeError, TypeError):
        return resp
    if not isinstance(parsed, dict):
        return resp

    column_fallbacks = {
        "precision": ("precision",),
        "recall": ("recall",),
        "f1_score": ("f1_score",),
        "accuracy": ("accuracy",),
    }
    for field, keys in column_fallbacks.items():
        if getattr(resp, field, None) is None:
            for k in keys:
                v = parsed.get(k)
                if v is not None:
                    setattr(resp, field, v)
                    break

    for key in _EXTRA_METRIC_KEYS:
        v = parsed.get(key)
        if v is not None:
            setattr(resp, key, v)
    return resp


def _load_dataset_names(db: Session, records: list[ModelRecord]) -> dict[int, str]:
    """Batch-resolve dataset.name для усіх dataset_id у records (1 SQL)."""
    ids = {r.dataset_id for r in records if r.dataset_id}
    if not ids:
        return {}
    rows = db.query(Dataset.id, Dataset.name).filter(Dataset.id.in_(ids)).all()
    return {dsid: name for dsid, name in rows}


@router.get("", response_model=List[ModelRecordResponse])
def list_models(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    records = db.query(ModelRecord).order_by(ModelRecord.created_at.desc()).all()
    name_map = _load_dataset_names(db, records)
    return [_serialize_model(r, name_map.get(r.dataset_id)) for r in records]


@router.patch("/{model_id}/activate", response_model=ModelRecordResponse)
def activate_model(
    model_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    target = db.query(ModelRecord).filter(ModelRecord.id == model_id).first()
    if not target:
        raise HTTPException(status_code=404, detail="Model not found")

    base_path = os.path.join(os.path.dirname(__file__), "../..")
    pkl_path = None
    if target.filename:
        pkl_path = os.path.join(base_path, "models", target.filename)
        if not os.path.exists(pkl_path):
            raise HTTPException(status_code=404, detail="Model .pkl file missing from disk")

    db.query(ModelRecord).update({"is_active": False})
    target.is_active = True
    db.commit()
    db.refresh(target)

    if pkl_path:
        from api.main import detector
        detector.load_from_file(pkl_path)

    ds_name = None
    if target.dataset_id:
        ds = db.query(Dataset.name).filter(Dataset.id == target.dataset_id).first()
        ds_name = ds[0] if ds else None
    return _serialize_model(target, ds_name)


@router.delete("", status_code=200)
def delete_all_models(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    models = db.query(ModelRecord).all()
    if not models:
        raise HTTPException(status_code=404, detail="No models to delete")

    base_path = os.path.join(os.path.dirname(__file__), "../..")
    models_dir = os.path.join(base_path, "models")

    for m in models:
        if not m.filename:
            continue
        pkl_path = os.path.join(models_dir, m.filename)
        if os.path.exists(pkl_path):
            os.remove(pkl_path)

    count = len(models)
    db.query(ModelRecord).delete()
    db.commit()

    return {"message": f"Видалено {count} моделей"}


@router.delete("/{model_id}", status_code=200)
def delete_model(
    model_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """Видалити одну модель + її .pkl з диска (для LLM presets — лише запис у БД)."""
    record = db.query(ModelRecord).filter(ModelRecord.id == model_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Model not found")

    if record.filename:
        base_path = os.path.join(os.path.dirname(__file__), "../..")
        pkl_path = os.path.join(base_path, "models", record.filename)
        if os.path.exists(pkl_path):
            try:
                os.remove(pkl_path)
            except OSError as e:
                logger.warning(f"Could not remove {pkl_path}: {e}")

    db.delete(record)
    db.commit()
    return {"message": "Модель видалено", "deleted_id": model_id}


@router.post("/{model_id}/evaluate")
def evaluate_model(
    model_id: int,
    req: EvaluateModelRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Прогнати модель на test split активного датасету, отримати метрики.

    Для NB/DistilBERT/GNN: використовує натреновану модель (Colab lazy-load by model_path).
    Для LLM: ЛОКАЛЬНО через api.llm_predictor.predict_batch_with_preset (async,
    повертає job_id; UI робить polling /models/llm-jobs/{job_id}).

    Метрики зберігаються у ModelRecord для майбутніх порівнянь.
    """
    record = db.query(ModelRecord).filter(ModelRecord.id == model_id).first()
    if not record:
        raise HTTPException(404, "Модель не знайдена")

    active_ds = db.query(Dataset).filter(
        Dataset.user_id == current_user.id,
        Dataset.is_active == True,
    ).first()
    if not active_ds:
        raise HTTPException(400, "Немає активного датасету. Активуйте у Datasets page.")

    if record.model_type == "llm":
        try:
            preset_config = json.loads(record.llm_config) if record.llm_config else {}
        except json.JSONDecodeError as e:
            raise HTTPException(500, f"Invalid llm_config JSON: {e}")

        job_id = create_job()
        user_id = current_user.id
        dataset_id = active_ds.id
        max_samples = req.max_samples
        splits_subdir = req.splits_subdir or "splits_cross_domain"

        def _run_llm_eval() -> None:
            try:
                metrics = evaluate_llm_preset(
                    user_id=user_id,
                    dataset_id=dataset_id,
                    preset_config=preset_config,
                    max_samples=max_samples,
                    splits_subdir=splits_subdir,
                    progress_callback=lambda c, t: update_progress(job_id, c, t),
                )
                predictions_compact = metrics.pop("_predictions_compact", None)
                with SessionLocal() as fresh_db:
                    fresh_record = (
                        fresh_db.query(ModelRecord)
                        .filter(ModelRecord.id == model_id)
                        .first()
                    )
                    if fresh_record:
                        fresh_record.accuracy = metrics.get("accuracy")
                        fresh_record.precision = metrics.get("precision")
                        fresh_record.recall = metrics.get("recall")
                        fresh_record.f1_score = metrics.get("f1_score")
                        fresh_record.metrics_json = json.dumps(metrics)
                        if predictions_compact:
                            fresh_record.predictions_json = json.dumps(predictions_compact)
                        fresh_db.commit()
                mark_done(job_id, metrics)
            except Exception as e:
                logger.exception(f"LLM evaluation failed for model {model_id}")
                mark_failed(job_id, str(e))

        background_tasks.add_task(_run_llm_eval)

        return {
            "job_id": job_id,
            "status": "started",
            "message": (
                "LLM evaluation запущено асинхронно. "
                "Опитуйте /models/llm-jobs/{job_id}"
            ),
        }

    is_colab = os.getenv("IS_COLAB", "false").lower() in ("true", "1", "t")
    if not is_colab:
        raise HTTPException(503, "Evaluation потребує Colab (IS_COLAB=true)")

    colab_base = os.getenv("COLAB_NGROK_URL", "").rstrip("/")
    if not colab_base:
        raise HTTPException(500, "COLAB_NGROK_URL не встановлено")

    if record.model_type in ("nb", "deberta", "distilbert", "bert", "gin", "sage", "gnn"):
        if not record.model_path:
            raise HTTPException(
                400,
                "Модель не має model_path. Перетренуйте модель перш ніж evaluate.",
            )

        if record.model_type in ("gin", "sage"):
            colab_model_type = "gnn"
            architecture: str | None = record.model_type
        elif record.model_type == "deberta":
            colab_model_type = "distilbert"
            architecture = None
        else:
            colab_model_type = record.model_type
            architecture = None

        payload = {
            "dataset_id": active_ds.id,
            "dataset_name": active_ds.name,
            "max_samples": req.max_samples,
            "test_size": req.test_size,
            "model_type": colab_model_type,
            "model_path": record.model_path,
        }
        if architecture:
            payload["model_params"] = {"architecture": architecture}

    else:
        raise HTTPException(400, f"Unknown model_type: {record.model_type}")

    async_url = f"{colab_base}/run_evaluation_async"
    status_base = f"{colab_base}/training_status"
    logger.info(f"Evaluating model_id={model_id} type={record.model_type} on ds={active_ds.id}")

    try:
        start_resp = requests.post(async_url, json=payload, timeout=60)
        start_resp.raise_for_status()
        job_id = start_resp.json().get("job_id")
        if not job_id:
            raise HTTPException(500, "No job_id returned")

        logger.info(f"Async evaluation started: {job_id}")

        import time as _time
        start_time = _time.time()
        MAX_WAIT = 14400
        POLL_INTERVAL = 5

        while True:
            if _time.time() - start_time > MAX_WAIT:
                raise HTTPException(504, f"Evaluation timeout. Job {job_id} still running.")

            try:
                sr = requests.get(f"{status_base}/{job_id}", timeout=30)
                sr.raise_for_status()
                sd = sr.json()
            except requests.exceptions.RequestException:
                _time.sleep(POLL_INTERVAL)
                continue

            if sd.get("status") == "done":
                result = sd.get("result", {})
                break
            elif sd.get("status") == "failed":
                raise HTTPException(500, f"Evaluation failed: {sd.get('error')}")

            _time.sleep(POLL_INTERVAL)
    except requests.exceptions.ConnectionError as e:
        raise HTTPException(503, f"Colab недоступний: {e}")
    except requests.exceptions.Timeout:
        raise HTTPException(504, "Evaluation timeout (>4 год)")
    except requests.exceptions.HTTPError:
        raise HTTPException(
            start_resp.status_code,
            f"Colab error: {start_resp.text[:500]}",
        )

    metrics = result.get("metrics", {})
    try:
        record.accuracy = metrics.get("accuracy")
        record.precision = metrics.get("precision")
        record.recall = metrics.get("recall")
        record.f1_score = metrics.get("f1_score")
        record.metrics_json = json.dumps(metrics)
        db.commit()
        logger.info(f"Saved metrics for model {model_id}: acc={metrics.get('accuracy')}")
    except Exception as e:
        logger.warning(f"Failed to save metrics: {e}")
        db.rollback()

    return {
        "model_id": model_id,
        "model_name": record.name,
        "model_type": record.model_type,
        "dataset_id": active_ds.id,
        "dataset_name": active_ds.name,
        "metrics": metrics,
        "samples": result.get("samples", []),
    }


@router.get("/llm-jobs/{job_id}")
def get_llm_job_status(
    job_id: str,
    _: User = Depends(get_current_user),
):
    """Polling статусу async LLM evaluation job (in-memory).

    Повертає:
        status: "pending" | "running" | "done" | "failed"
        progress: [current, total]
        result: dict з метриками (тільки якщо status="done")
        error: рядок (тільки якщо status="failed")
    """
    status = get_status(job_id)
    if status is None:
        raise HTTPException(404, f"Job {job_id} not found")
    return status
