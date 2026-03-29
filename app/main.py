# app/main.py
import logging
import uuid
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.model import FakeNewsModel
from app.features import compute_features
from app.ensemble import apply_voting
from app.schemas import (
    PredictRequest as PredictRequestV2,
    NBConfig, XLMRConfig, LLMConfig,
)
from app.database import create_tables, get_db, Experiment, ModelRecord, User
from app.auth import get_current_user
from app.routers import auth as auth_router
from app.routers import experiments as experiments_router
from app.routers import models_router
from sqlalchemy.orm import Session
import os
import requests
from app.text_preprocessing import preprocess_for_bayes, preprocess_for_transformer

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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
create_tables()
app.include_router(auth_router.router)
app.include_router(experiments_router.router)
app.include_router(models_router.router)

# 3. Схеми для /predict — імпортовані з app.schemas (PredictRequestV2)




class TrainRequest(BaseModel):
    mode: str = "single"          # "single" | "ensemble"
    models: list[dict] = []       # [{model: "nb", variant: "complement", vectorizer: "tfidf", ...}]
    ensemble: dict | None = None
    preprocessing: dict | None = None

@app.get("/")

def read_root():
    return {"status": "System is running", "model_loaded": True}

# ── Predict helpers ───────────────────────────────────────────────────────────

def _run_model(model_cfg, text: str, metadata: dict | None = None) -> dict:
    """
    Запустити одну модель і повернути результат:
    {'model', 'label', 'probability', 'feature_values'}.
    """
    mtype = model_cfg.model

    # Compute additional features if configured
    additional = model_cfg.additional_features
    feature_values = None
    meta_dict = metadata.model_dump() if metadata else None
    if additional and additional.mask:
        feature_values = compute_features(text, additional.mask, meta_dict)

    af_dict = {"groups": additional.groups, "mask": additional.mask} if additional else None

    # ── Naive Bayes: local prediction ─────────────────────────────────────
    if mtype == 'nb':
        bayes_text = preprocess_for_bayes(text)
        result = detector.predict(bayes_text, additional_features=af_dict, metadata=meta_dict)
        return {
            'model': mtype,
            'label': result['label'],
            'probability': result['score'],
            'feature_values': feature_values,
        }

    # ── XLM-RoBERTa: Colab GPU inference ─────────────────────────────────
    if mtype == 'xlm_r':
        transformer_text = preprocess_for_transformer(text)
        colab_url = os.environ.get("COLAB_NGROK_URL", "").strip()
        if colab_url:
            payload = {
                'text': transformer_text,
                'integration_mode': getattr(model_cfg, 'integration_mode', 'concat'),
            }
            if additional:
                payload['additional_features'] = af_dict
            if meta_dict:
                payload['metadata'] = meta_dict
            try:
                resp = requests.post(
                    f"{colab_url.rstrip('/')}/predict_xlm",
                    json=payload,
                    timeout=120,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    prob = data.get('confidence') or data.get('probability')
                    label = data.get('label', 'FAKE' if data.get('is_fake') else 'REAL')
                    if isinstance(label, int):
                        label = 'FAKE' if label == 1 else 'REAL'
                    return {
                        'model': mtype,
                        'label': label,
                        'probability': float(prob) if prob is not None else None,
                        'feature_values': feature_values,
                    }
            except Exception as e:
                logger.warning(f"XLM-R Colab inference failed: {e}")
        # Fallback to local NB detector
        result = detector.predict(preprocess_for_bayes(text), additional_features=af_dict, metadata=meta_dict)
        return {
            'model': mtype,
            'label': result['label'],
            'probability': result['score'],
            'feature_values': feature_values,
        }

    # ── LLM: local Gemini API call ─────────────────────────────────────────
    if mtype == 'llm':
        from app.llm_predictor import predict as llm_predict
        llm_mode = getattr(model_cfg, 'mode', 'single')
        try:
            llm_result = llm_predict(preprocess_for_transformer(text), mode=llm_mode, feature_values=feature_values)
            return {
                'model': mtype,
                'label': llm_result['label'],
                'probability': llm_result['confidence'],
                'feature_values': feature_values,
                'reason': llm_result.get('reason', ''),
            }
        except Exception as e:
            logger.error(f"LLM prediction failed: {e}")
            raise HTTPException(status_code=503, detail=f"LLM prediction failed: {e}")

    raise HTTPException(status_code=400, detail=f"Unknown model type: {mtype}")


# ── POST /predict ─────────────────────────────────────────────────────────────

@app.post("/predict")
def predict_news(
    request: PredictRequestV2,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Класифікація тексту: одна модель або ансамбль.

    Якщо `models` не передано — використовується активна локальна модель.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text field cannot be empty")

    input_text = request.text

    # Якщо список моделей порожній — зворотна сумісність зі старим клієнтом
    if not request.models:
        result = detector.predict(preprocess_for_bayes(input_text))
        return {
            "mode": "single",
            "is_fake": result["label"] == "FAKE",
            "confidence": result["score"],
            "label": 1 if result["label"] == "FAKE" else 0,
            "features": result.get("details", {}),
        }

    # Run all models (pass metadata for social features)
    individual_results = [_run_model(cfg, input_text, request.metadata) for cfg in request.models]

    if request.mode == "single":
        r = individual_results[0]
        is_fake = r['label'] == 'FAKE'
        return {
            "mode": "single",
            "is_fake": is_fake,
            "confidence": r['probability'] or 0.5,
            "label": 1 if is_fake else 0,
            "features": {},
            "feature_values": r.get('feature_values'),
            "individual_results": individual_results,
        }

    # Ensemble voting
    ensemble_cfg = request.ensemble
    strategy = ensemble_cfg.strategy if ensemble_cfg else "soft"
    weights = ensemble_cfg.weights if ensemble_cfg else None
    ensemble_result = apply_voting(individual_results, strategy, weights)

    return {
        "mode": "ensemble",
        "is_fake": ensemble_result["label"] == "FAKE",
        "confidence": ensemble_result["confidence"],
        "label": 1 if ensemble_result["label"] == "FAKE" else 0,
        "ensemble_result": ensemble_result,
        "individual_results": individual_results,
        "features": {},
    }



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


@app.post("/train")
def train_model(
    request: TrainRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Проксує тренування до Colab через ngrok. Датасет завантажується в Colab."""
    global detector

    colab_url = os.environ.get("COLAB_NGROK_URL", "").strip()
    if not colab_url:
        raise HTTPException(
            status_code=503,
            detail="Colab ML server not configured. Set COLAB_NGROK_URL.",
        )

    experiment_id = str(uuid.uuid4())

    # Extract NB config from first model in the list
    nb_cfg = request.models[0] if request.models else {}
    model_type = "ensemble" if request.mode == "ensemble" else "naive_bayes"

    payload = {
        "user_id": str(current_user.id),
        "experiment_id": experiment_id,
        "model_type": model_type,
        "nb_variant": nb_cfg.get("variant", "multinomial"),
        "vectorizer_type": nb_cfg.get("vectorizer", "tfidf"),
        "ngram_range": nb_cfg.get("ngram_range", "1,1"),
        "alpha": float(nb_cfg.get("alpha", 1.0)),
        "features": nb_cfg.get("additional_features") or {},
        "preprocessing": request.preprocessing or {},
    }

    try:
        resp = requests.post(
            f"{colab_url.rstrip('/')}/run_training",
            json=payload,
            timeout=300,
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"Colab request failed: {e}")
        fail_record = Experiment(
            experiment_id=experiment_id,
            user_id=current_user.id,
            model_type=model_type,
            status="failed",
        )
        db.add(fail_record)
        db.commit()
        raise HTTPException(
            status_code=503,
            detail="Втрачено зв'язок з ML-сервером. Перевірте, чи запущений Colab.",
        ) from e

    if resp.status_code != 200:
        try:
            err = resp.json()
            msg = err.get("message", err.get("error", resp.text))
        except Exception:
            msg = resp.text
        raise HTTPException(status_code=502, detail=msg)

    data = resp.json()
    if data.get("status") != "success":
        raise HTTPException(
            status_code=502,
            detail=data.get("message", "Training failed"),
        )

    download_url = data.get("download_url")
    if not download_url:
        raise HTTPException(
            status_code=502,
            detail="Colab did not return download_url",
        )

    base_path = os.path.join(os.path.dirname(__file__), "..")
    models_dir = os.path.join(base_path, "models")
    os.makedirs(models_dir, exist_ok=True)
    local_path = os.path.join(models_dir, f"trained_model_{experiment_id}.pkl")

    try:
        _download_model_from_url(download_url, local_path)
    except Exception as e:
        logger.error(f"Model download failed: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"Не вдалося завантажити модель: {e}",
        ) from e

    detector.load_from_file(local_path)
    logger.info(f"Model loaded from {local_path}")

    metrics = data.get("metrics", {})
    pkl_filename = f"trained_model_{experiment_id}.pkl"

    # Write experiment record to DB
    experiment_record = Experiment(
        experiment_id=experiment_id,
        user_id=current_user.id,
        model_type=model_type,
        model_file=pkl_filename,
        accuracy=metrics.get("accuracy"),
        precision=metrics.get("precision"),
        recall=metrics.get("recall"),
        f1_score=metrics.get("f1_score"),
        train_size=metrics.get("train_size"),
        test_size=metrics.get("test_size"),
        training_time=str(metrics.get("training_time", "")),
        status="success",
    )
    db.add(experiment_record)

    # Deactivate old active model, register new one as active
    db.query(ModelRecord).update({"is_active": False})
    model_record = ModelRecord(
        experiment_id=experiment_id,
        filename=pkl_filename,
        model_type=model_type,
        accuracy=metrics.get("accuracy"),
        is_active=True,
    )
    db.add(model_record)
    db.commit()

    return {
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1_score": metrics.get("f1_score"),
        "confusion_matrix": metrics.get("confusion_matrix"),
        "train_size": metrics.get("train_size"),
        "test_size": metrics.get("test_size"),
        "training_time": metrics.get("training_time"),
    }



# Для запуску: uvicorn app.main:app --reload