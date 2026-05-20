
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import json
import logging
import uuid
import os
from datetime import datetime, timezone
import requests
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from api.model import FakeNewsModel
from api.schemas import ModelConfig
from api.database import create_tables, get_db, Experiment, ModelRecord, User, Dataset
from api.auth import get_current_user
from api.routers import auth as auth_router
from api.routers import models_router
from api.routers import sources as sources_router
from api.routers import llm_presets as llm_presets_router
from api.routers import datasets as datasets_router
from api.routers import analytics as analytics_router
from api.routers import ensembles as ensembles_router
from api.routers import analyze_v2 as analyze_v2_router

from api.text_preprocessing import preprocess_for_bayes, preprocess_for_transformer
from api.fact_check import verify_post

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fake News Detection API",
    description="Бекенд для системи виявлення дезінформації на основі BERT",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = FakeNewsModel()

os.makedirs("uploaded_datasets", exist_ok=True)
create_tables()

if not os.getenv("GOOGLE_FACT_CHECK_API_KEY"):
    logger.warning(
        "GOOGLE_FACT_CHECK_API_KEY not set — fact-check stage буде повертати "
        "{found:false, error:'GOOGLE_FACT_CHECK_API_KEY not configured'}"
    )
app.include_router(auth_router.router)
app.include_router(models_router.router)
app.include_router(sources_router.router)
app.include_router(datasets_router.router)
app.include_router(llm_presets_router.router)
app.include_router(analytics_router.router)
app.include_router(ensembles_router.router)
app.include_router(analyze_v2_router.router)


from api.ml_client import MLServerError, MLServerNotReadyError, MLServerOfflineError
from fastapi.requests import Request as _FastAPIRequest
from fastapi.responses import JSONResponse as _JSONResponse


@app.exception_handler(MLServerError)
def _ml_server_error_handler(_request: _FastAPIRequest, exc: MLServerError):
    """Уніфікований структурований response для ML offline / not ready."""
    code = (
        "ml_server_offline"
        if isinstance(exc, MLServerOfflineError)
        else "ml_server_not_ready"
        if isinstance(exc, MLServerNotReadyError)
        else "ml_server_error"
    )
    user_msg = (
        "Colab ML server недоступний. "
        "Перевір Colab notebook і онови ML_SERVER_URL (або COLAB_NGROK_URL) у .env"
        if code == "ml_server_offline"
        else "Colab ML server недоступний"
    )
    return _JSONResponse(
        status_code=exc.status_code,
        content={
            "error": code,
            "message": user_msg,
            "detail": exc.message,
            "checked_url": exc.checked_url,
        },
    )

FEATURE_GROUP_KEYS: dict[str, list[str]] = {
    "emotional": [
        "sentiment_score", "emotion_intensity", "emoji_count", "exclamation_count",
        "anger_score", "fear_score", "anticipation_score", "trust_score",
        "surprise_score", "sadness_score", "joy_score", "disgust_score",
        "positive_score", "negative_score",
    ],
    "stylistic": ["caps_ratio", "ttr", "repetition_score", "avg_word_length"],
    "rhetorical": ["clickbait_score", "authority_refs", "pronoun_ratio", "question_count"],
}


class TrainRequest(BaseModel):
    experiment_id: str | None = "default_experiment"
    model_name: str | None = None
    model_type: str | None = None
    model_params: dict | None = {}

    mode: str = "single"
    models: list[ModelConfig] = []
    ensemble: dict | None = None
    preprocessing: dict | None = {}
    

@app.get("/")

def read_root():
    return {"status": "System is running", "model_loaded": True}


@app.get("/health")
def health_check():
    """Liveness self-probe. ML-сервер перевіряти НЕ обов'язково — для цього є
    `/ml-server/status`. Тут лише FastAPI alive-сигнал для k8s/uptime probes."""
    return {"status": "ok", "service": "fake_news_api"}


@app.get("/ml-server/status")
def ml_server_status(force: bool = False):
    """Cached health-check Colab ML server. Використовується FE banner-ом.

    Query: `?force=true` обходить 30s cache.
    """
    from api.ml_client import check_status
    return check_status(force=force)


class AnalyzeRequest(BaseModel):
    text: str
    model_id: int
    explain: bool = False


@app.post("/analyze")
def analyze_text(
    request: AnalyzeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    record = db.query(ModelRecord).filter(ModelRecord.id == request.model_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Модель не знайдена")

    mtype = record.model_type
    from api import ml_client
    colab_url = ml_client.colab_url() or ""

    if record.pipeline_type == "aggregated" and mtype == "deberta":
        colab_url = ml_client.ensure_healthy()
        article_text = _fetch_article_for_aggregated(request.text)
        formatted = _format_aggregated_inference_text(request.text, article_text)
        payload: dict = {"text": formatted}
        if record.model_path:
            payload["model_path"] = record.model_path
        try:
            resp = requests.post(f"{colab_url.rstrip('/')}/predict_deberta", json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            prob = data.get("probability", 0.5)
            return {
                "label": data["label"],
                "confidence": data.get("confidence", abs(prob - 0.5) * 2),
                "probability": prob,
            }
        except requests.exceptions.ConnectionError:
            raise HTTPException(status_code=503, detail="Colab недоступний. Перевірте COLAB_NGROK_URL.")
        except requests.exceptions.HTTPError:
            raise HTTPException(status_code=resp.status_code, detail=f"Colab error: {resp.text[:500]}")

    if mtype == "nb":
        colab_status = ml_client.check_status()
        try_colab = colab_status["ok"] and bool(record.model_path)
        if try_colab:
            try:
                payload = {
                    "text": request.text,
                    "model_path": record.model_path,
                    "explain": bool(request.explain),
                }
                resp = requests.post(
                    f"{colab_url.rstrip('/')}/predict_nb",
                    json=payload,
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    nb_response: dict = {
                        "label": data["label"],
                        "confidence": data["confidence"],
                        "probability": data["probability"],
                    }
                    if "explanation" in data:
                        nb_response["explanation"] = data["explanation"]
                    if "explanation_error" in data:
                        nb_response["explanation_error"] = data["explanation_error"]
                    return nb_response

                ctype = (resp.headers.get("content-type") or "").lower()
                if resp.status_code == 404 and "text/html" in ctype:
                    logger.warning(
                        "Colab /predict_nb returned HTML 404 — endpoint missing. "
                        "Falling back to local detector (`git pull` ml_server на Colab?)"
                    )
                else:
                    try:
                        err = resp.json()
                    except ValueError:
                        err = {}
                    err_code = err.get("error")
                    msg = err.get("message") or resp.text[:300]
                    if err_code == "empty_text":
                        raise HTTPException(400, "Текст порожній")
                    if err_code in ("pkl_not_found", "model_path_required"):
                        raise HTTPException(
                            404,
                            f"NB модель не знайдена на Colab диску ({err_code})",
                        )
                    if err_code == "wrong_model_type":
                        raise HTTPException(
                            400,
                            f"Bundle на Colab — не NB: {err.get('got')}",
                        )
                    if err_code in ("aggregated_unsupported", "features_required"):
                        raise HTTPException(400, msg)
                    if err_code == "predict_failed":
                        raise HTTPException(502, f"Colab predict_nb помилка: {msg}")
                    raise HTTPException(
                        502, f"Colab /predict_nb error: {resp.text[:300]}"
                    )
            except requests.exceptions.ConnectionError:
                ml_client.invalidate_cache()
        result = detector.predict(preprocess_for_bayes(request.text), use_text=True)
        prob = result["score"]
        nb_response = {
            "label": result["label"],
            "confidence": abs(prob - 0.5) * 2,
            "probability": prob,
        }
        if request.explain:
            expl = detector.explain_nb_prediction(request.text)
            if expl is not None:
                nb_response["explanation"] = expl
        return nb_response

    if mtype in ("distilbert", "deberta"):
        colab_url = ml_client.ensure_healthy()
        try:
            payload = {"text": preprocess_for_transformer(request.text)}
            if record.model_path:
                payload["model_path"] = record.model_path
            resp = requests.post(
                f"{colab_url.rstrip('/')}/predict_distilbert",
                json=payload,
                timeout=120,
            )
            if resp.status_code == 200:
                data = resp.json()
                prob = data.get("probability", 0.5)
                distil_response: dict = {
                    "label": data["label"],
                    "confidence": data.get("confidence", abs(prob - 0.5) * 2),
                    "probability": prob,
                }
                if request.explain:
                    try:
                        expl_payload = {"text": preprocess_for_transformer(request.text)}
                        if record.model_path:
                            expl_payload["model_path"] = record.model_path
                        expl_resp = requests.post(
                            f"{colab_url.rstrip('/')}/explain_distilbert",
                            json=expl_payload,
                            timeout=60,
                        )
                        if expl_resp.status_code == 200:
                            distil_response["explanation"] = expl_resp.json()
                        else:
                            logger.warning(
                                "explain_distilbert returned %s: %s",
                                expl_resp.status_code, expl_resp.text[:200],
                            )
                    except Exception as e:
                        logger.warning(f"explain_distilbert proxy failed: {e}")
                return distil_response
            try:
                err = resp.json()
            except ValueError:
                err = {}
            err_code = err.get("error")
            msg = err.get("message") or resp.text[:300]
            if err_code == "empty_text":
                raise HTTPException(
                    status_code=400,
                    detail="Текст порожній після extraction",
                )
            if err_code == "model_not_loaded":
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "DistilBERT модель не завантажена на Colab. "
                        "Перетренуй модель або перезапусти Colab notebook."
                    ),
                )
            if err_code in (
                "pkl_not_found",
                "model_dir_not_found",
                "incomplete_model_dir",
            ):
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Файли моделі не знайдено на Colab диску ({err_code}). "
                        "Найімовірніше runtime restart втратив локальний кеш — "
                        "перетренуй або перезавантаж з Drive."
                    ),
                )
            if err_code in ("wrong_model_type", "missing_model_dir"):
                raise HTTPException(
                    status_code=500,
                    detail=f"Bundle .pkl некоректний ({err_code}): {msg}",
                )
            if err_code == "model_load_failed":
                raise HTTPException(
                    status_code=502,
                    detail=f"Не вдалося завантажити модель: {msg}",
                )
            raise HTTPException(
                status_code=502, detail=f"Colab error: {resp.text[:500]}"
            )
        except requests.exceptions.ConnectionError:
            raise HTTPException(
                status_code=503,
                detail="Colab недоступний. Перевірте COLAB_NGROK_URL.",
            )

    if mtype in ("gnn", "gin", "sage"):
        colab_url = ml_client.ensure_healthy()
        if not record.model_path:
            raise HTTPException(status_code=400, detail="GNN модель не має model_path")

        import asyncio as _asyncio
        from api.inference_context import (
            build_inference_context,
            derive_requirements,
        )

        if record.inference_requirements:
            try:
                inference_reqs = json.loads(record.inference_requirements)
            except (TypeError, ValueError):
                inference_reqs = derive_requirements(
                    model_type=mtype, pipeline_type=record.pipeline_type,
                )
        else:
            inference_reqs = derive_requirements(
                model_type=mtype, pipeline_type=record.pipeline_type,
            )

        try:
            context = _asyncio.run(build_inference_context(
                text=request.text, inference_requirements=inference_reqs,
            ))
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"Не вдалося побудувати propagation context: {e}",
            )

        graph_inputs = context.get("graph_data") or {}
        if not graph_inputs.get("tweets"):
            return {
                "label": "UNKNOWN",
                "confidence": 0.0,
                "probability": 0.5,
                "error": "no_propagation_graph",
                "message": (
                    "Не знайдено релевантних постів у соцмережах. GIN/SAGE "
                    "моделі потребують каскаду поширення для inference."
                ),
                "inference_context": {
                    "claim": context.get("claim"),
                    "metadata": context.get("metadata"),
                },
            }

        gnn_payload: dict = {
            "model_path": record.model_path,
            "graph_inputs": {
                "article_text": graph_inputs["article_text"],
                "tweets": graph_inputs["tweets"],
                "retweets": graph_inputs["retweets"],
                "replies": graph_inputs["replies"],
            },
            "explain": bool(request.explain),
        }
        try:
            resp = requests.post(
                f"{colab_url.rstrip('/')}/predict_gnn",
                json=gnn_payload,
                timeout=120,
            )
            if resp.status_code == 200:
                data = resp.json()
                prob = data.get("probability", data.get("proba_fake", 0.5))
                result: dict = {
                    "label": data["label"],
                    "confidence": data.get("confidence", abs(prob - 0.5) * 2),
                    "probability": prob,
                    "graph_stats": data.get("graph_stats") or {
                        "n_nodes": data.get("graph_size"),
                        "n_edges": (data.get("graph_edges") or 0) * 2,
                    },
                    "architecture": data.get("architecture"),
                    "inference_context": {
                        "claim": context.get("claim"),
                        "propagation_stats": (graph_inputs.get("metadata") or {}),
                        "metadata": context.get("metadata"),
                    },
                }
                if "explanation" in data:
                    result["explanation"] = data["explanation"]
                if "explanation_error" in data:
                    result["explanation_error"] = data["explanation_error"]
                return result
            raise HTTPException(status_code=502, detail=f"Colab error: {resp.text[:300]}")
        except requests.exceptions.ConnectionError:
            raise HTTPException(status_code=503, detail="Colab недоступний")

    if mtype == "llm":
        from api.llm_predictor import predict_with_preset

        if not record.llm_config:
            raise HTTPException(
                status_code=400,
                detail="LLM model record has no config. Create a preset first via POST /llm-presets",
            )
        try:
            preset_config = json.loads(record.llm_config)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=500, detail=f"Invalid llm_config JSON: {e}")

        try:
            r = predict_with_preset(
                preprocess_for_transformer(request.text),
                preset_config,
            )
            if r["label"] == "UNCERTAIN":
                return {
                    "label": "UNCERTAIN",
                    "confidence": 0.0,
                    "probability": None,
                    "reason": r.get("reason", ""),
                    "base_model_used": r.get("base_model_used", ""),
                    "mode": r.get("mode", ""),
                }
            prob = r["confidence"] if r["label"] == "FAKE" else 1.0 - r["confidence"]
            return {
                "label": r["label"],
                "confidence": r["confidence"],
                "probability": prob,
                "reason": r.get("reason", ""),
                "base_model_used": r.get("base_model_used", ""),
                "mode": r.get("mode", ""),
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"LLM error: {e}")

    raise HTTPException(status_code=400, detail=f"Unknown model: {mtype}")


class FactCheckRequest(BaseModel):
    text: str
    model_label: str | None = None
    model_confidence: float | None = None
    language: str = "en"


@app.post("/fact_check")
def fact_check_endpoint(
    payload: FactCheckRequest,
    current_user: User = Depends(get_current_user),
):
    """Single post verification — fact-check claim + compare with model output."""
    return verify_post(
        post_text=payload.text,
        model_label=payload.model_label,
        model_confidence=payload.model_confidence,
        language=payload.language,
    )


class FactCheckBatchPost(BaseModel):
    id: str | None = None
    text: str
    model_label: str | None = None
    model_confidence: float | None = None


class FactCheckBatchRequest(BaseModel):
    posts: list[FactCheckBatchPost]
    language: str = "en"


@app.post("/fact_check_batch")
def fact_check_batch_endpoint(
    payload: FactCheckBatchRequest,
    current_user: User = Depends(get_current_user),
):
    """Batch verification with summary stats."""
    results = []
    for post in payload.posts:
        result = verify_post(
            post_text=post.text,
            model_label=post.model_label,
            model_confidence=post.model_confidence,
            language=payload.language,
        )
        result["id"] = post.id
        results.append(result)

    summary = {
        "total": len(results),
        "fact_check_found": sum(1 for r in results if r["fact_check_found"]),
        "match": sum(1 for r in results if r["comparison_status"] == "MATCH"),
        "mismatch": sum(1 for r in results if r["comparison_status"] == "MISMATCH"),
        "no_data": sum(1 for r in results if r["comparison_status"] == "NO_DATA"),
        "mixed": sum(1 for r in results if r["comparison_status"] == "MIXED"),
    }
    comparable = summary["match"] + summary["mismatch"]
    summary["accuracy"] = summary["match"] / comparable if comparable > 0 else None

    return {"results": results, "summary": summary}


class TrainNBArticleRequest(BaseModel):
    """Параметри для article-level NB pipeline (тонкий проксі до Colab)."""

    model_name: str | None = None
    use_emotional: bool = True
    use_stylistic: bool = True
    use_rhetorical: bool = True
    use_social: bool = True
    tfidf_max_features: int = 5000
    random_seed: int = 42


def _normalize_confusion_matrix(cm) -> dict | None:
    """Accept either [[tn,fp],[fn,tp]] (sklearn list) or {tn,fp,fn,tp} (dict from Colab)."""
    if cm is None:
        return None
    if isinstance(cm, dict):
        keys = {"tn", "fp", "fn", "tp"}
        if keys.issubset(cm.keys()):
            try:
                return {k: int(cm[k]) for k in keys}
            except (TypeError, ValueError):
                return None
        return None
    if isinstance(cm, (list, tuple)) and len(cm) == 2 and len(cm[0]) == 2:
        return {
            "tn": int(cm[0][0]),
            "fp": int(cm[0][1]),
            "fn": int(cm[1][0]),
            "tp": int(cm[1][1]),
        }
    return None


@app.post("/train_nb_article")
def train_nb_article_endpoint(
    request: TrainNBArticleRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Article-level NB training через Colab `/train_nb_article`.

    Returns flattened metrics + confusion_matrix у форматі, який очікує
    існуючий UI (TrainResponse у `ui/src/types.ts`).
    """
    active_ds = db.query(Dataset).filter(
        Dataset.user_id == current_user.id,
        Dataset.is_active == True,
    ).first()
    if not active_ds:
        raise HTTPException(
            status_code=400,
            detail="Немає активного датасету. Активуйте у Datasets page.",
        )

    colab_base = os.getenv("COLAB_NGROK_URL", "").rstrip("/")
    if not colab_base:
        raise HTTPException(status_code=500, detail="COLAB_NGROK_URL не встановлено")

    from api.colab_sync import ensure_dataset_on_colab, ColabSyncError
    try:
        ensure_dataset_on_colab(
            dataset_id=active_ds.id,
            dataset_folder=active_ds.folder_path,
        )
    except ColabSyncError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Не вдалося синхронізувати dataset з Colab: {e}",
        )

    payload = {
        "dataset_id": active_ds.id,
        "dataset_name": active_ds.name,
        "use_emotional": request.use_emotional,
        "use_stylistic": request.use_stylistic,
        "use_rhetorical": request.use_rhetorical,
        "use_social": request.use_social,
        "tfidf_max_features": request.tfidf_max_features,
        "random_seed": request.random_seed,
    }

    target_url = f"{colab_base}/train_nb_article"
    logger.info(f"Article-level NB training → {target_url} (ds={active_ds.id})")

    try:
        resp = requests.post(target_url, json=payload, timeout=600)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"Colab недоступний: {e}")
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Article training timeout (>10хв)")
    except requests.exceptions.HTTPError:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Colab error: {resp.text[:500]}",
        )

    result = resp.json()
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=f"Training failed: {result}")

    metrics = result.get("metrics", {}) or {}
    model_dir = result.get("model_dir")

    name = (
        request.model_name
        or f"Article-NB · ds{active_ds.id} · {datetime.now(timezone.utc).strftime('%m%d-%H%M')}"
    )
    base_filename = os.path.basename(model_dir) if model_dir else None
    if base_filename:
        filename = base_filename
        if db.query(ModelRecord).filter(ModelRecord.filename == filename).first():
            filename = f"{filename}_{int(datetime.now(timezone.utc).timestamp())}"
    else:
        filename = f"nb_article_{int(datetime.now(timezone.utc).timestamp())}"

    data_stats_nba = result.get("data_stats", {}) or {}
    splits_used_nba = data_stats_nba.get("splits_used") or active_ds.active_split

    predictions_compact_nba = result.get("predictions_compact")
    predictions_json_nba = (
        json.dumps(predictions_compact_nba) if predictions_compact_nba else None
    )

    record = ModelRecord(
        name=name,
        model_type="nb",
        pipeline_type="article",
        filename=filename,
        model_path=model_dir,
        accuracy=metrics.get("accuracy"),
        precision=metrics.get("precision"),
        recall=metrics.get("recall"),
        f1_score=metrics.get("f1_score"),
        metrics_json=json.dumps(metrics),
        predictions_json=predictions_json_nba,
        splits_used=splits_used_nba,
        dataset_id=active_ds.id,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    cm = _normalize_confusion_matrix(metrics.get("confusion_matrix"))
    return {
        "status": "success",
        "path": model_dir,
        "model_id": record.id,
        "feature_count": result.get("feature_count"),
        "metrics": {
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1_score": metrics.get("f1_score"),
            "f1_macro": metrics.get("f1_macro"),
            "roc_auc": metrics.get("roc_auc"),
            "train_size": metrics.get("n_train"),
            "test_size": metrics.get("n_test"),
            "confusion_matrix": cm,
        },
    }


class PredictNBArticleRequest(BaseModel):
    title: str = ""
    text: str = ""
    model_id: int


@app.post("/predict_nb_article")
def predict_nb_article_endpoint(
    request: PredictNBArticleRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not (request.title.strip() or request.text.strip()):
        raise HTTPException(status_code=400, detail="title або text не може бути порожнім")

    record = db.query(ModelRecord).filter(
        ModelRecord.id == request.model_id,
        ModelRecord.pipeline_type == "article",
    ).first()
    if not record:
        raise HTTPException(status_code=404, detail="Article-level model не знайдена")
    if not record.model_path:
        raise HTTPException(status_code=400, detail="Модель без model_path — перетренуйте")

    colab_base = os.getenv("COLAB_NGROK_URL", "").rstrip("/")
    if not colab_base:
        raise HTTPException(status_code=500, detail="COLAB_NGROK_URL не встановлено")

    target_url = f"{colab_base}/predict_nb_article"
    payload = {
        "title": request.title,
        "text": request.text,
        "model_dir": record.model_path,
    }

    try:
        resp = requests.post(target_url, json=payload, timeout=60)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"Colab недоступний: {e}")
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Inference timeout")
    except requests.exceptions.HTTPError:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Colab error: {resp.text[:500]}",
        )

    return resp.json()


AGGREGATED_SOCIAL_FEATURES = [
    "tweet_count", "mean_followers", "mean_friends",
    "verified_ratio", "mean_statuses", "mean_account_age_days",
    "mean_retweets", "mean_favorites",
]


class TrainAggregatedRequest(BaseModel):
    """Aggregated pipeline: 1 representative tweet + social aggregates + article."""

    model_name: str | None = None
    model_type: str = "nb"

    use_text: bool = True
    use_social_aggregates: bool = True
    use_emotional: bool = False
    use_stylistic: bool = False
    use_rhetorical: bool = False

    nb_variant: str = "complement"
    tfidf_max_features: int = 5000
    alpha: float = 1.0

    test_ratio: float = 0.20
    seed: int = 42
    top_tweet_strategy: str = "popularity"


@app.post("/train_aggregated")
def train_aggregated_endpoint(
    request: TrainAggregatedRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    @deprecated
    Aggregated pipeline — feeds the ML server with `pipeline_mode='aggregated'`,
    routed through `/run_training_async` (same async path as /train).

    Не використовується з UI: NB і DistilBERT тепер тренуються через /train
    (article-level pipeline). Endpoint лишено для backward compat зі старими
    aggregated моделями, що вже є у БД.
    """
    if request.model_type not in ("nb", "deberta"):
        raise HTTPException(status_code=400, detail="model_type повинен бути 'nb' або 'deberta'")

    active_ds = db.query(Dataset).filter(
        Dataset.user_id == current_user.id,
        Dataset.is_active == True,
    ).first()
    if not active_ds:
        raise HTTPException(
            status_code=400,
            detail="Немає активного датасету. Активуйте у Datasets page.",
        )

    is_colab = os.getenv("IS_COLAB", "false").lower() in ("true", "1", "t")
    if is_colab:
        colab_base = os.getenv("COLAB_NGROK_URL", "").rstrip("/")
        if not colab_base:
            raise HTTPException(status_code=500, detail="Увімкнено IS_COLAB, але COLAB_NGROK_URL порожній")
        ml_base = colab_base
    else:
        ml_base = "http://127.0.0.1:5050"

    from api.colab_sync import ensure_dataset_on_colab, ColabSyncError
    try:
        ensure_dataset_on_colab(
            dataset_id=active_ds.id,
            dataset_folder=active_ds.folder_path,
        )
    except ColabSyncError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Не вдалося синхронізувати dataset з Colab: {e}",
        )

    feature_mask: dict = {}
    if request.model_type == "nb":
        if request.use_emotional:
            for k in FEATURE_GROUP_KEYS["emotional"]:
                feature_mask[k] = True
        if request.use_stylistic:
            for k in FEATURE_GROUP_KEYS["stylistic"]:
                feature_mask[k] = True
        if request.use_rhetorical:
            for k in FEATURE_GROUP_KEYS["rhetorical"]:
                feature_mask[k] = True

    social_features = list(AGGREGATED_SOCIAL_FEATURES) if request.use_social_aggregates else []

    if request.model_type == "nb":
        model_params = {
            "nb_variant": request.nb_variant,
            "vectorizer_type": "tfidf",
            "alpha": request.alpha,
            "tfidf_max_features": request.tfidf_max_features,
            "additional_features": {
                "mask": feature_mask,
                "social_extra": social_features,
            },
        }
    else:
        model_params = {}

    from api.utils.experiment_naming import generate_experiment_id
    agg_experiment_id = "agg_" + generate_experiment_id(
        model_type=request.model_type,
        model_params=model_params,
        splits_subdir=None,
        custom_name=request.model_name,
    )
    logger.info(f"Aggregated experiment_id: {agg_experiment_id}")

    payload = {
        "user_id": current_user.id,
        "experiment_id": agg_experiment_id,
        "model_type": request.model_type,
        "model_params": model_params,
        "use_text": request.use_text,
        "dataset_id": active_ds.id,
        "dataset_name": active_ds.name,
        "data_params": {
            "test_ratio": request.test_ratio,
            "seed": request.seed,
            "top_tweet_strategy": request.top_tweet_strategy,
            "social_aggregate_features": social_features,
        },
        "model_name": request.model_name,
    }

    async_url = f"{ml_base}/run_training_async"
    status_base = f"{ml_base}/training_status"
    logger.info(f"Aggregated training → {async_url} (ds={active_ds.id}, mtype={request.model_type})")

    try:
        start_response = requests.post(async_url, json=payload, timeout=60)
        start_response.raise_for_status()
        start_data = start_response.json()
        job_id = start_data.get("job_id")
        if not job_id:
            raise ValueError(f"No job_id in response: {start_data}")
    except requests.exceptions.ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"ML-сервер: {type(e).__name__}: {str(e)[:300]}")
    except requests.exceptions.HTTPError:
        raise HTTPException(status_code=start_response.status_code, detail=f"ML error: {start_response.text[:500]}")

    import time as _time
    POLL_INTERVAL = 5
    MAX_WAIT = 14400
    start_time = _time.time()
    last_progress = None
    ml_data: dict = {}

    while True:
        elapsed = _time.time() - start_time
        if elapsed > MAX_WAIT:
            raise HTTPException(status_code=504, detail=f"Aggregated training timeout (>{MAX_WAIT}s). Job {job_id} may still be running.")

        try:
            status_response = requests.get(f"{status_base}/{job_id}", timeout=30)
            status_response.raise_for_status()
            status_data = status_response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Polling error (will retry): {e}")
            _time.sleep(POLL_INTERVAL)
            continue

        current_status = status_data.get("status")
        current_progress = status_data.get("progress")
        if current_progress != last_progress:
            logger.info(f"[{int(elapsed)}s] Job {job_id[:8]} status={current_status} progress={current_progress}")
            last_progress = current_progress

        if current_status == "done":
            ml_data = status_data.get("result", {}) or {}
            break
        if current_status == "failed":
            error_msg = status_data.get("error", "Unknown error")
            logger.error(f"Aggregated training failed: {error_msg}")
            raise HTTPException(status_code=500, detail=f"Тренування провалилось: {error_msg[:500]}")

        _time.sleep(POLL_INTERVAL)

    metrics = ml_data.get("metrics", {}) or {}
    model_path = ml_data.get("path")

    name = (
        request.model_name
        or f"Aggregated-{request.model_type.upper()} · ds{active_ds.id} · {datetime.now(timezone.utc).strftime('%m%d-%H%M')}"
    )
    base_filename = os.path.basename(model_path) if model_path else None
    if base_filename:
        filename = base_filename
        if db.query(ModelRecord).filter(ModelRecord.filename == filename).first():
            filename = f"{filename}_{int(datetime.now(timezone.utc).timestamp())}"
    else:
        filename = f"agg_{request.model_type}_{int(datetime.now(timezone.utc).timestamp())}"

    data_stats_agg = ml_data.get("data_stats", {}) or {}
    splits_used_agg = data_stats_agg.get("splits_used") or active_ds.active_split

    predictions_compact_agg = ml_data.get("predictions_compact")
    predictions_json_agg = (
        json.dumps(predictions_compact_agg) if predictions_compact_agg else None
    )

    record = ModelRecord(
        name=name,
        model_type=request.model_type,
        pipeline_type="aggregated",
        filename=filename,
        model_path=model_path,
        accuracy=metrics.get("accuracy"),
        precision=metrics.get("precision"),
        recall=metrics.get("recall"),
        f1_score=metrics.get("f1_score"),
        metrics_json=json.dumps(metrics),
        predictions_json=predictions_json_agg,
        splits_used=splits_used_agg,
        dataset_id=active_ds.id,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    cm = _normalize_confusion_matrix(metrics.get("confusion_matrix"))
    return {
        "status": "success",
        "path": model_path,
        "model_id": record.id,
        "metrics": {
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1_score": metrics.get("f1_score"),
            "f1_macro": metrics.get("f1_macro"),
            "roc_auc": metrics.get("roc_auc"),
            "train_size": metrics.get("train_size") or metrics.get("n_train"),
            "test_size": metrics.get("test_size") or metrics.get("n_test"),
            "training_time": metrics.get("training_time"),
            "confusion_matrix": cm,
        },
        "feature_samples": ml_data.get("feature_samples", []),
        "top_words": ml_data.get("top_words", {}),
        "data_stats": ml_data.get("data_stats", {}),
    }


class PredictAggregatedRequest(BaseModel):
    text: str
    model_id: int


def _format_aggregated_inference_text(text: str, article_text: str = "") -> str:
    """Build the [TWEET]/[ARTICLE] format the aggregated model expects.

    [SOCIAL] block removed — соц.ознаки тепер числові, передаються окремо
    через payload['aggregates'] на ML-сервер. Якщо frontend їх не надає,
    ML-сервер сам підставить train_mean.
    """
    return f"[TWEET] {text} [ARTICLE] {article_text}"


def _fetch_article_for_aggregated(text: str) -> str:
    """If text contains a URL, try to fetch its article body. Empty string on failure."""
    import re
    m = re.search(r"https?://\S+", text)
    if not m:
        return ""
    try:
        from api.fact_check import fetch_article_text
        return fetch_article_text(m.group(0)) or ""
    except Exception:
        return ""


@app.post("/predict_aggregated")
def predict_aggregated_endpoint(
    request: PredictAggregatedRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text не може бути порожнім")

    record = db.query(ModelRecord).filter(ModelRecord.id == request.model_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Модель не знайдена")
    if record.pipeline_type != "aggregated":
        raise HTTPException(status_code=400, detail="Модель не aggregated типу")

    colab_base = os.getenv("COLAB_NGROK_URL", "").rstrip("/")
    if not colab_base:
        raise HTTPException(status_code=500, detail="COLAB_NGROK_URL не встановлено")

    article_text = _fetch_article_for_aggregated(request.text)
    formatted = _format_aggregated_inference_text(request.text, article_text)

    endpoint = "/predict_nb" if record.model_type == "nb" else "/predict_deberta"
    payload: dict = {"text": formatted}
    if record.model_path:
        payload["model_path"] = record.model_path

    try:
        resp = requests.post(f"{colab_base}{endpoint}", json=payload, timeout=60)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"Colab недоступний: {e}")
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Inference timeout")
    except requests.exceptions.HTTPError:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Colab error: {resp.text[:500]}",
        )

    return resp.json()


def _download_model_from_url(url: str, dest_path: str) -> None:
    """Завантажити .pkl з URL (Drive або інший)."""
    if "drive.google.com" in url:
        try:
            import gdown
            gdown.download(url, dest_path, quiet=True, fuzzy=True)
        except Exception as e:
            logger.warning(f"gdown failed: {e}, falling back to requests")
            r = requests.get(url, timeout=60, stream=True)
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    else:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(r.content)


import os
import requests
from fastapi import HTTPException

@app.post("/train")
def train_model(
    request: TrainRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    print("--- ОТРИМАНО ЗАПИТ ВІД ФРОНТЕНДУ ---")
    print(request.model_dump())
    print("------------------------------------")

    actual_model_type = request.model_type
    if not actual_model_type and request.models and len(request.models) > 0:
        actual_model_type = request.models[0].model

    if not actual_model_type:
        raise HTTPException(status_code=400, detail="Не вказано тип моделі для тренування")

    gnn_architecture = None
    if actual_model_type == "gnn":
        params = request.model_params or {}
        gnn_architecture = params.get("architecture")
        if gnn_architecture not in ("gin", "sage"):
            raise HTTPException(
                status_code=400,
                detail="Для model_type='gnn' треба вказати model_params.architecture='gin' або 'sage'",
            )

    model_params = request.model_params or {}
    if request.models and len(request.models) > 0:
        from_models = request.models[0].model_dump(exclude={"model"})
        model_params = {**from_models, **model_params}
    logger.info(f"model_params sent to Colab: {model_params}")

    is_colab = os.getenv("IS_COLAB", "false").lower() in ("true", "1", "t")
    
    if is_colab:
        colab_base = os.getenv("COLAB_NGROK_URL", "").rstrip("/")
        if not colab_base:
            raise HTTPException(status_code=500, detail="Увімкнено IS_COLAB, але COLAB_NGROK_URL порожній")
        
        target_url = f"{colab_base}/run_training"
        print(f"Маршрутизація: Google Colab -> {target_url}")
    else:
        local_base = "http://127.0.0.1:5050"
        target_url = f"{local_base}/run_training"
        print(f"Маршрутизація: Local Flask -> {target_url}")

    active_ds = db.query(Dataset).filter(
        Dataset.user_id == current_user.id,
        Dataset.is_active == True,
    ).first()
    if not active_ds:
        raise HTTPException(
            status_code=400,
            detail="Немає активного датасету. Активуйте у Datasets page.",
        )

    from api.colab_sync import ensure_dataset_on_colab, ColabSyncError
    try:
        sync_result = ensure_dataset_on_colab(
            dataset_id=active_ds.id,
            dataset_folder=active_ds.folder_path,
        )
        if not sync_result.get("skipped"):
            logger.info(f"Re-synced: {sync_result.get('chunks_sent')} chunks")
    except ColabSyncError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Не вдалося синхронізувати dataset з Colab: {e}"
        )

    splits_subdir = f"splits_{active_ds.active_split}" if active_ds.active_split else None

    from api.utils.experiment_naming import (
        generate_experiment_id,
        is_default_experiment_id,
    )
    experiment_id = request.experiment_id
    if is_default_experiment_id(experiment_id):
        experiment_id = generate_experiment_id(
            model_type=actual_model_type,
            model_params=model_params,
            splits_subdir=splits_subdir,
            custom_name=request.model_name,
        )
        logger.info(f"Auto-generated experiment_id: {experiment_id}")

    payload = {
        "user_id": current_user.id,
        "experiment_id": experiment_id,
        "model_type": actual_model_type,
        "model_params": model_params,
        "preprocessing": request.preprocessing or {},
        "dataset_id": active_ds.id,
        "dataset_name": active_ds.name,
        "data_params": {
            "splits_subdir": splits_subdir,
        },
    }
    print(f"DEBUG: Sending dataset_id={active_ds.id}, name={active_ds.name!r} (no CSV in payload)")

    print(f"Відправка на ML-сервер: {target_url}")

    try:
        payload_keys = list(payload.keys())
        print(f"DEBUG target_url={target_url!r}")
        print(f"DEBUG payload keys: {payload_keys}")
        for k, v in payload.items():
            if isinstance(v, str) and len(v) > 100:
                print(f"DEBUG payload[{k}] = <str {len(v):,} chars>")
            else:
                print(f"DEBUG payload[{k}] = {v!r}")
        total_size = sum(len(str(v)) if not isinstance(v, (dict, list)) else len(json.dumps(v)) for v in payload.values())
        print(f"DEBUG total payload size: ~{total_size:,} chars ({total_size/1024/1024:.2f} MB)")

        async_url = target_url.replace("/run_training", "/run_training_async")
        status_base = target_url.rsplit("/", 1)[0] + "/training_status"

        print(f"Запускаю async тренування: {async_url}")
        start_response = requests.post(async_url, json=payload, timeout=60)
        start_response.raise_for_status()
        start_data = start_response.json()
        job_id = start_data.get("job_id")
        if not job_id:
            raise ValueError(f"No job_id in response: {start_data}")

        print(f"✓ Job started: {job_id}")
        logger.info(f"Async training job started: {job_id}")

        import time as _time
        POLL_INTERVAL = 5
        MAX_WAIT = 14400
        start_time = _time.time()
        last_progress = None

        while True:
            elapsed = _time.time() - start_time
            if elapsed > MAX_WAIT:
                raise HTTPException(
                    status_code=504,
                    detail=f"Training timeout (>{MAX_WAIT}s). Job {job_id} may still be running on Colab."
                )

            try:
                status_response = requests.get(
                    f"{status_base}/{job_id}",
                    timeout=30,
                )
                status_response.raise_for_status()
                status_data = status_response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Polling error (will retry): {e}")
                _time.sleep(POLL_INTERVAL)
                continue

            current_status = status_data.get("status")
            current_progress = status_data.get("progress")

            if current_progress != last_progress:
                print(f"[{int(elapsed)}s] Job {job_id[:8]} status={current_status} progress={current_progress}")
                last_progress = current_progress

            if current_status == "done":
                ml_data = status_data.get("result", {})
                if not ml_data:
                    raise ValueError("Job done but no result returned")
                logger.info(f"✓ Async training complete after {int(elapsed)}s")
                break

            elif current_status == "failed":
                error_msg = status_data.get("error", "Unknown error")
                tb = status_data.get("traceback", "")
                logger.error(f"Async training failed: {error_msg}\n{tb}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Тренування провалилось: {error_msg[:500]}"
                )

            _time.sleep(POLL_INTERVAL)

        logger.info(f"ML response keys: {list(ml_data.keys())}, has top_words: {'top_words' in ml_data}")

    except requests.exceptions.ConnectionError as e:
        print(f"DEBUG ConnectionError type: {type(e).__name__}")
        print(f"DEBUG ConnectionError: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=503,
            detail=f"ML-сервер: {type(e).__name__}: {str(e)[:300]}"
        )
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Час очікування вичерпано")
    except requests.exceptions.HTTPError:
        print(f"DEBUG HTTPError, response status: {response.status_code}")
        print(f"DEBUG response body: {response.text[:500]}")
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Помилка ML-сервера: {response.text[:500]}"
        )
    except Exception as e:
        print(f"DEBUG Other: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise

    metrics = ml_data.get("metrics", {})
    exp_id = experiment_id
    base_exp_id = exp_id
    counter = 0
    while db.query(Experiment).filter(Experiment.experiment_id == exp_id).first():
        counter += 1
        exp_id = f"{base_exp_id}_{counter}"

    stored_model_type = gnn_architecture if gnn_architecture else actual_model_type

    if actual_model_type in ("nb", "distilbert"):
        stored_pipeline_type = "article"
    elif actual_model_type == "gnn":
        stored_pipeline_type = "graph"
    else:
        stored_pipeline_type = "tweet"

    exp = Experiment(
        experiment_id=exp_id,
        user_id=current_user.id,
        dataset_id=active_ds.id,
        model_type=stored_model_type,
        model_file=ml_data.get("path"),
        accuracy=metrics.get("accuracy"),
        precision=metrics.get("precision"),
        recall=metrics.get("recall"),
        f1_score=metrics.get("f1_score"),
        train_size=metrics.get("train_size"),
        test_size=metrics.get("test_size"),
        training_time=str(metrics.get("training_time", "")),
        status="success",
        model_configs=json.dumps([m.model_dump() for m in request.models]) if request.models else None,
    )
    db.add(exp)

    model_file = ml_data.get("path")
    if model_file:
        base_filename = os.path.basename(model_file)
        filename = f"{exp_id}_{base_filename}"

        if db.query(ModelRecord).filter(ModelRecord.filename == filename).first():
            import time as _time
            collided = filename
            filename = f"{filename}_{int(_time.time())}"
            logger.warning(f"Filename collision: {collided} → {filename}")

        data_stats = ml_data.get("data_stats", {}) or {}
        splits_used = data_stats.get("splits_used") or active_ds.active_split
        mr_metrics_json = json.dumps(metrics) if metrics else None
        predictions_compact = ml_data.get("predictions_compact")
        predictions_json_str = (
            json.dumps(predictions_compact) if predictions_compact else None
        )
        new_record = ModelRecord(
            experiment_id=exp_id,
            filename=filename,
            name=request.model_name or None,
            model_path=model_file,
            model_type=stored_model_type,
            pipeline_type=stored_pipeline_type,
            accuracy=metrics.get("accuracy"),
            precision=metrics.get("precision"),
            recall=metrics.get("recall"),
            f1_score=metrics.get("f1_score"),
            metrics_json=mr_metrics_json,
            predictions_json=predictions_json_str,
            splits_used=splits_used,
            dataset_id=active_ds.id,
        )
        db.add(new_record)
        db.flush()
        logger.info(
            f"Created ModelRecord id={new_record.id}, "
            f"filename={filename}, type={stored_model_type}"
        )

    db.commit()

    return ml_data