# api/main.py

# Load .env BEFORE any imports that read environment variables (auth, llm_predictor, etc.)
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
from api.routers import verification as verification_router
from api.routers import analytics as analytics_router

from api.text_preprocessing import preprocess_for_bayes, preprocess_for_transformer
from api.fact_check import verify_post

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 1. Ініціалізація додатку
app = FastAPI(
    title="Fake News Detection API",
    description="Бекенд для системи виявлення дезінформації на основі BERT",
    version="1.0.0"
)

# CORS налаштування для React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Ініціалізація моделі (завантажується один раз при старті)
# Ми робимо це глобально, щоб не вантажити модель при кожному запиті
detector = FakeNewsModel()

# Ініціалізація БД та підключення роутерів
os.makedirs("uploaded_datasets", exist_ok=True)
create_tables()
app.include_router(auth_router.router)
app.include_router(models_router.router)
app.include_router(sources_router.router)
app.include_router(datasets_router.router)
app.include_router(llm_presets_router.router)
app.include_router(verification_router.router)
app.include_router(analytics_router.router)

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

    # Старі поля від фронтенду
    mode: str = "single"
    models: list[ModelConfig] = []
    ensemble: dict | None = None
    preprocessing: dict | None = {}
    

@app.get("/")

def read_root():
    return {"status": "System is running", "model_loaded": True}


# ── POST /analyze ─────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text: str
    model_id: int  # ModelRecord.id


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
    colab_url = os.environ.get("COLAB_NGROK_URL", "").strip()

    # Aggregated pipeline reformats the input as [TWEET]/[SOCIAL]/[ARTICLE]
    # and dispatches to /predict_deberta. NB more не використовує aggregated:
    # нові article-level NB моделі йдуть напряму у /predict_nb нижче, а старі
    # aggregated NB записи інференсяться так само на сирому тексті.
    if record.pipeline_type == "aggregated" and mtype == "deberta":
        if not colab_url:
            raise HTTPException(status_code=503, detail="COLAB_NGROK_URL не встановлено")
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
        if colab_url:
            try:
                payload = {"text": request.text}
                if record.model_path:
                    payload["model_path"] = record.model_path
                resp = requests.post(
                    f"{colab_url.rstrip('/')}/predict_nb",
                    json=payload,
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        "label": data["label"],
                        "confidence": data["confidence"],
                        "probability": data["probability"],
                    }
                raise HTTPException(status_code=502, detail=f"Colab error: {resp.text}")
            except requests.exceptions.ConnectionError:
                raise HTTPException(status_code=503, detail="Colab недоступний. Перевірте COLAB_NGROK_URL.")
        # Fallback: local model
        result = detector.predict(preprocess_for_bayes(request.text), use_text=True)
        prob = result["score"]
        return {
            "label": result["label"],
            "confidence": abs(prob - 0.5) * 2,
            "probability": prob,
        }

    if mtype == "distilbert":
        if not colab_url:
            raise HTTPException(status_code=503, detail="COLAB_NGROK_URL не встановлено")
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
                return {
                    "label": data["label"],
                    "confidence": data.get("confidence", abs(prob - 0.5) * 2),
                    "probability": prob,
                }
            raise HTTPException(status_code=502, detail=f"Colab error: {resp.text}")
        except requests.exceptions.ConnectionError:
            raise HTTPException(status_code=503, detail="Colab недоступний. Перевірте COLAB_NGROK_URL.")

    # Legacy: old records still tagged "deberta" — route to /predict_distilbert
    # (Colab keeps it as the canonical endpoint; old /predict_deberta is deprecated).
    if mtype == "deberta":
        if not colab_url:
            raise HTTPException(status_code=503, detail="COLAB_NGROK_URL не встановлено")
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
                return {
                    "label": data["label"],
                    "confidence": data.get("confidence", abs(prob - 0.5) * 2),
                    "probability": prob,
                }
            raise HTTPException(status_code=502, detail=f"Colab error: {resp.text}")
        except requests.exceptions.ConnectionError:
            raise HTTPException(status_code=503, detail="Colab недоступний. Перевірте COLAB_NGROK_URL.")

    if mtype in ("gnn", "gin", "sage"):
        if not colab_url:
            raise HTTPException(status_code=503, detail="COLAB_NGROK_URL не встановлено")
        if not record.model_path:
            raise HTTPException(status_code=400, detail="GNN модель не має model_path")

        # Простий випадок: тільки article_text. Tweets/retweets/replies можна додати
        # пізніше, якщо UI підтримуватиме контекст соц.мережі.
        gnn_payload: dict = {
            "article_text": request.text,
            "model_path": record.model_path,
            "tweets": [],
            "retweets": [],
            "replies": [],
        }
        try:
            resp = requests.post(
                f"{colab_url.rstrip('/')}/predict_gnn",
                json=gnn_payload,
                timeout=120,
            )
            if resp.status_code == 200:
                data = resp.json()
                prob = data.get("proba_fake", 0.5)
                return {
                    "label": data["label"],
                    "confidence": data.get("confidence", abs(prob - 0.5) * 2),
                    "probability": prob,
                    "graph_size": data.get("graph_size"),
                    "graph_edges": data.get("graph_edges"),
                    "architecture": data.get("architecture"),
                }
            raise HTTPException(status_code=502, detail=f"Colab error: {resp.text}")
        except requests.exceptions.ConnectionError:
            raise HTTPException(status_code=503, detail="Colab недоступний")

    if mtype == "llm":
        from api.llm_predictor import predict_with_preset

        # LLM presets store config as JSON in record.llm_config
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


# ── POST /fact_check ──────────────────────────────────────────────────────────

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


# ── POST /train_nb_article ────────────────────────────────────────────────────


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
    # Already a dict from ML server (most common path now)
    if isinstance(cm, dict):
        keys = {"tn", "fp", "fn", "tp"}
        if keys.issubset(cm.keys()):
            try:
                return {k: int(cm[k]) for k in keys}
            except (TypeError, ValueError):
                return None
        return None
    # List format [[tn,fp],[fn,tp]] from sklearn confusion_matrix().tolist()
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

    # Lazy sync — якщо Colab рестартував, re-upload
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

    # Save ModelRecord with pipeline_type='article'
    name = (
        request.model_name
        or f"Article-NB · ds{active_ds.id} · {datetime.now(timezone.utc).strftime('%m%d-%H%M')}"
    )
    filename = os.path.basename(model_dir) if model_dir else None
    if filename and db.query(ModelRecord).filter(ModelRecord.filename == filename).first():
        filename = f"{filename}_{int(datetime.now(timezone.utc).timestamp())}"

    data_stats_nba = result.get("data_stats", {}) or {}
    splits_used_nba = data_stats_nba.get("splits_used") or active_ds.active_split

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
        splits_used=splits_used_nba,
        dataset_id=active_ds.id,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    # Flat shape for UI's TrainResponse
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


# ── POST /predict_nb_article ──────────────────────────────────────────────────


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


# ── POST /train_aggregated ────────────────────────────────────────────────────


AGGREGATED_SOCIAL_FEATURES = [
    "tweet_count", "mean_followers", "mean_friends",
    "verified_ratio", "mean_statuses", "mean_account_age_days",
    "mean_retweets", "mean_favorites",
]


class TrainAggregatedRequest(BaseModel):
    """Aggregated pipeline: 1 representative tweet + social aggregates + article."""

    model_name: str | None = None
    model_type: str = "nb"  # "nb" | "deberta"

    use_text: bool = True
    use_social_aggregates: bool = True
    use_emotional: bool = False
    use_stylistic: bool = False
    use_rhetorical: bool = False

    # NB hyperparams
    nb_variant: str = "complement"           # "complement" | "multinomial"
    tfidf_max_features: int = 5000
    alpha: float = 1.0

    # Common
    test_ratio: float = 0.20
    seed: int = 42
    top_tweet_strategy: str = "popularity"   # "popularity" | "first" | "random"


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

    # Lazy sync — same as /train
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

    # Build feature mask for additional NB features
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

    payload = {
        "user_id": current_user.id,
        "experiment_id": "aggregated",
        "model_type": request.model_type,
        "model_params": model_params,
        "use_text": request.use_text,
        "dataset_id": active_ds.id,
        "dataset_name": active_ds.name,
        "data_params": {
            "test_ratio": request.test_ratio,
            "seed": request.seed,
            "top_tweet_strategy": request.top_tweet_strategy,
            # Соц.агрегати йдуть як числові ознаки в pipeline:
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

    # ── Poll until done (same loop as /train) ──
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

    # Persist ModelRecord with pipeline_type='aggregated'
    name = (
        request.model_name
        or f"Aggregated-{request.model_type.upper()} · ds{active_ds.id} · {datetime.now(timezone.utc).strftime('%m%d-%H%M')}"
    )
    filename = os.path.basename(model_path) if model_path else None
    if filename and db.query(ModelRecord).filter(ModelRecord.filename == filename).first():
        filename = f"{filename}_{int(datetime.now(timezone.utc).timestamp())}"

    data_stats_agg = ml_data.get("data_stats", {}) or {}
    splits_used_agg = data_stats_agg.get("splits_used") or active_ds.active_split

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
        # Forward to UI — already has rendering logic for both:
        "feature_samples": ml_data.get("feature_samples", []),
        "top_words": ml_data.get("top_words", {}),
        "data_stats": ml_data.get("data_stats", {}),
    }


# ── POST /predict_aggregated ──────────────────────────────────────────────────


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
        from api.fact_check import fetch_article_text  # may not exist — handled below
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
    # У майбутньому тут можна передати реальні агрегати з Bluesky/Mastodon API:
    # payload["aggregates"] = {"tweet_count": ..., "mean_followers": ..., ...}
    # Зараз empty — ML-сервер підставить train_mean як fallback.

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

    # 1. Визначаємо тип моделі (з прямого поля або з масиву models)
    actual_model_type = request.model_type
    if not actual_model_type and request.models and len(request.models) > 0:
        actual_model_type = request.models[0].model

    if not actual_model_type:
        raise HTTPException(status_code=400, detail="Не вказано тип моделі для тренування")

    # GNN merger: фронтенд шле model_type="gnn" + model_params.architecture="gin"|"sage".
    # У БД зберігаємо model_type як architecture (gin/sage), щоб ModelsPage показав
    # правильну іконку. На Colab летить "gnn" як було.
    gnn_architecture = None
    if actual_model_type == "gnn":
        params = request.model_params or {}
        gnn_architecture = params.get("architecture")
        if gnn_architecture not in ("gin", "sage"):
            raise HTTPException(
                status_code=400,
                detail="Для model_type='gnn' треба вказати model_params.architecture='gin' або 'sage'",
            )

    # 2. Формуємо параметри моделі — завжди беремо з models[0] якщо є
    model_params = request.model_params or {}
    if request.models and len(request.models) > 0:
        from_models = request.models[0].model_dump(exclude={"model"})
        model_params = {**from_models, **model_params}  # model_params overrides if both set
    logger.info(f"model_params sent to Colab: {model_params}")

    # 3. Визначаємо цільовий URL залежно від режиму (Local vs Colab)
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

    # 4. Find active dataset for current user
    active_ds = db.query(Dataset).filter(
        Dataset.user_id == current_user.id,
        Dataset.is_active == True,
    ).first()
    if not active_ds:
        raise HTTPException(
            status_code=400,
            detail="Немає активного датасету. Активуйте у Datasets page.",
        )

    # 5. Lazy sync — якщо Colab рестартував, re-upload
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

    # 6. Simple payload — no CSV data
    splits_subdir = f"splits_{active_ds.active_split}" if active_ds.active_split else None

    payload = {
        "user_id": current_user.id,
        "experiment_id": request.experiment_id or "default_exp",
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
        # DEBUG
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

        # ── ASYNC PATTERN: start job, then poll ──
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

        # ── Polling loop ──
        import time as _time
        POLL_INTERVAL = 5  # seconds
        MAX_WAIT = 14400   # 4 hours
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

    # Save experiment to DB
    metrics = ml_data.get("metrics", {})
    exp_id = request.experiment_id or "default_exp"
    # Ensure unique experiment_id (append suffix if collision)
    base_exp_id = exp_id
    counter = 0
    while db.query(Experiment).filter(Experiment.experiment_id == exp_id).first():
        counter += 1
        exp_id = f"{base_exp_id}_{counter}"

    # Stored model_type: для GNN зберігаємо architecture (gin/sage), для інших — як є.
    stored_model_type = gnn_architecture if gnn_architecture else actual_model_type

    # pipeline_type: nb/distilbert тренуються на article-level, gnn — на graph;
    # старі aggregated моделі лишаються як є у БД (їхні записи не зачіпаються).
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

    # Register model file
    model_file = ml_data.get("path")
    if model_file:
        filename = os.path.basename(model_file)
        if not db.query(ModelRecord).filter(ModelRecord.filename == filename).first():
            data_stats = ml_data.get("data_stats", {}) or {}
            splits_used = data_stats.get("splits_used") or active_ds.active_split
            mr_metrics_json = json.dumps(metrics) if metrics else None
            db.add(ModelRecord(
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
                splits_used=splits_used,
                dataset_id=active_ds.id,
            ))
    db.commit()

    return ml_data