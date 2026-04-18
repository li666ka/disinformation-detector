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
import requests
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from api.model import FakeNewsModel
from api.ensemble import apply_voting
from api.schemas import (
    PredictRequest as PredictRequestV2,
    NBConfig, DeBERTaConfig, LLMConfig,
)
from api.database import create_tables, get_db, Experiment, ModelRecord, User, Dataset
from api.auth import get_current_user
from api.routers import auth as auth_router
from api.routers import experiments as experiments_router
from api.routers import models_router
from api.routers import sources as sources_router
from api.routers import llm_presets as llm_presets_router
from api.routers import datasets as datasets_router

from api.text_preprocessing import preprocess_for_bayes, preprocess_for_transformer

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
app.include_router(experiments_router.router)
app.include_router(models_router.router)
app.include_router(sources_router.router)
app.include_router(datasets_router.router)
app.include_router(llm_presets_router.router)

# 3. Схеми для /predict — імпортовані з api.schemas (PredictRequestV2)

class TrainRequest(BaseModel):
    experiment_id: str | None = "default_experiment"
    model_name: str | None = None
    model_type: str | None = None
    model_params: dict | None = {}

    # Старі поля від фронтенду
    mode: str = "single"
    models: list[dict] = []
    ensemble: dict | None = None
    preprocessing: dict | None = {}
    

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

    additional = model_cfg.additional_features
    meta_dict = metadata.model_dump() if metadata else None

    # ── Naive Bayes: Colab inference (consistent features) ────────────────
    if mtype == 'nb':
        colab_url = os.environ.get("COLAB_NGROK_URL", "").strip()
        if colab_url:
            mask = additional.mask if additional else {}
            feature_list = [k for k, v in mask.items() if v and k != "text"]
            try:
                resp = requests.post(
                    f"{colab_url.rstrip('/')}/predict_nb",
                    json={"text": text, "feature_list": feature_list},
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        'model': mtype,
                        'label': data['label'],
                        'probability': data['probability'],
                        'feature_values': data.get('feature_values'),
                    }
                logger.warning(f"Colab /predict_nb returned {resp.status_code}: {resp.text}")
            except Exception as e:
                logger.warning(f"NB Colab inference failed: {e}")
        # Fallback: local model (no feature extraction — Colab unavailable)
        bayes_text = preprocess_for_bayes(text)
        use_text = additional.mask.get("text", True) if additional else True
        result = detector.predict(bayes_text, feature_values=None, use_text=use_text)
        return {
            'model': mtype,
            'label': result['label'],
            'probability': result['score'],
            'feature_values': None,
        }

    # ── DeBERTa: Colab GPU inference ─────────────────────────────────
    if mtype == 'deberta':
        transformer_text = preprocess_for_transformer(text)
        colab_url = os.environ.get("COLAB_NGROK_URL", "").strip()
        print(colab_url)
        if colab_url:
            payload = {
                'text': transformer_text,
                'integration_mode': getattr(model_cfg, 'integration_mode', 'concat'),
            }
            if meta_dict:
                payload['metadata'] = meta_dict
            try:
                resp = requests.post(
                    f"{colab_url.rstrip('/')}/predict_deberta",
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
                        'feature_values': None,
                    }
            except Exception as e:
                logger.warning(f"DeBERTa Colab inference failed: {e}")
        # Fallback to local NB detector
        result = detector.predict(preprocess_for_bayes(text), feature_values=None, use_text=True)
        return {
            'model': mtype,
            'label': result['label'],
            'probability': result['score'],
            'feature_values': None,
        }

    # ── LLM: preset-driven Gemini call ─────────────────────────────────────
    if mtype == 'llm':
        from api.llm_predictor import predict_with_preset

        preset_id = getattr(model_cfg, 'preset_id', None)
        if preset_id is not None:
            # Load preset from DB
            from api.database import SessionLocal
            session = SessionLocal()
            try:
                record = session.query(ModelRecord).filter(
                    ModelRecord.id == preset_id,
                    ModelRecord.model_type == "llm",
                ).first()
                if not record or not record.llm_config:
                    raise HTTPException(
                        status_code=404,
                        detail=f"LLM preset id={preset_id} not found or has no config",
                    )
                preset_config = json.loads(record.llm_config)
            finally:
                session.close()
        else:
            # Inline default config (backward compat)
            llm_mode = getattr(model_cfg, 'mode', 'zero_shot')
            preset_config = {
                "base_model": "gemini-2.5-flash-lite",
                "mode": "bagging" if llm_mode == "bagging" else "zero_shot",
                "temperature": 0.7 if llm_mode == "bagging" else 0.0,
                "max_output_tokens": 200,
                "bagging_n_calls": 3,
            }

        try:
            llm_result = predict_with_preset(
                preprocess_for_transformer(text),
                preset_config,
            )
            label = llm_result['label']
            confidence = llm_result['confidence']
            if label == "UNCERTAIN":
                probability = None
            else:
                probability = confidence if label == "FAKE" else 1.0 - confidence
            return {
                'model': mtype,
                'label': label,
                'probability': probability,
                'feature_values': None,
                'reason': llm_result.get('reason', ''),
                'base_model_used': llm_result.get('base_model_used', ''),
                'mode': llm_result.get('mode', ''),
            }
        except Exception as e:
            logger.error(f"LLM prediction failed: {e}")
            raise HTTPException(status_code=503, detail=f"LLM prediction failed: {e}")

    raise HTTPException(status_code=400, detail=f"Unknown model type: {mtype}")


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

    if mtype == "deberta":
        if not colab_url:
            raise HTTPException(status_code=503, detail="COLAB_NGROK_URL не встановлено")
        try:
            resp = requests.post(
                f"{colab_url.rstrip('/')}/predict_deberta",
                json={"text": preprocess_for_transformer(request.text)},
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

    # Extract top discriminative words if any NB model is used
    has_nb = any(cfg.model == 'nb' for cfg in request.models)
    top_words = detector.get_top_words(n=20) if has_nb else None

    if request.mode == "single":
        r = individual_results[0]
        is_fake = r['label'] == 'FAKE'
        resp = {
            "mode": "single",
            "is_fake": is_fake,
            "confidence": r['probability'] or 0.5,
            "label": 1 if is_fake else 0,
            "features": {},
            "feature_values": r.get('feature_values'),
            "individual_results": individual_results,
        }
        if top_words:
            resp["top_words"] = top_words
        return resp

    # Ensemble voting
    ensemble_cfg = request.ensemble
    strategy = ensemble_cfg.strategy if ensemble_cfg else "soft"
    weights = ensemble_cfg.weights if ensemble_cfg else None
    ensemble_result = apply_voting(individual_results, strategy, weights)

    resp = {
        "mode": "ensemble",
        "is_fake": ensemble_result["label"] == "FAKE",
        "confidence": ensemble_result["confidence"],
        "label": 1 if ensemble_result["label"] == "FAKE" else 0,
        "ensemble_result": ensemble_result,
        "individual_results": individual_results,
        "features": {},
    }
    if top_words:
        resp["top_words"] = top_words
    return resp



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
        actual_model_type = request.models[0].get("model")

    if not actual_model_type:
        raise HTTPException(status_code=400, detail="Не вказано тип моделі для тренування")

    # 2. Формуємо параметри моделі — завжди беремо з models[0] якщо є
    model_params = request.model_params or {}
    if request.models and len(request.models) > 0:
        from_models = {k: v for k, v in request.models[0].items() if k != "model"}
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
            detail="Немає активного датасету. Завантажте датасет і зробіть його активним.",
        )

    news_csv_path = os.path.join(active_ds.folder_path, "news.csv")
    if not os.path.exists(news_csv_path):
        raise HTTPException(
            status_code=500,
            detail=f"news.csv не знайдено у датасеті: {news_csv_path}",
        )

    import base64
    with open(news_csv_path, "rb") as f:
        news_csv_b64 = base64.b64encode(f.read()).decode("ascii")

    # 5. Готуємо Payload для ML-сервера
    payload = {
        "user_id": current_user.id,
        "experiment_id": request.experiment_id or "default_exp",
        "model_type": actual_model_type,
        "model_params": model_params,
        "preprocessing": request.preprocessing or {},
        "dataset_id": active_ds.id,
        "dataset_name": active_ds.name,
        "news_csv_b64": news_csv_b64,
    }

    # Optional: send tweets.csv and users.csv for social features
    if active_ds.has_tweets:
        tweets_path = os.path.join(active_ds.folder_path, "tweets.csv")
        if os.path.exists(tweets_path):
            with open(tweets_path, "rb") as f:
                payload["tweets_csv_b64"] = base64.b64encode(f.read()).decode("ascii")

    if active_ds.has_users:
        users_path = os.path.join(active_ds.folder_path, "users.csv")
        if os.path.exists(users_path):
            with open(users_path, "rb") as f:
                payload["users_csv_b64"] = base64.b64encode(f.read()).decode("ascii")

    print(f"Відправка на ML-сервер: {target_url}")

    try:
        # Встановлюємо великий таймаут (4 години) для GPU-обчислень
        response = requests.post(target_url, json=payload, timeout=14400)
        response.raise_for_status()
        ml_data = response.json()
        logger.info(f"ML response keys: {list(ml_data.keys())}, has top_words: {'top_words' in ml_data}")
        
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503, 
            detail=f"ML-сервер недоступний за адресою {target_url}. Перевірте тунель або порт."
        )
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504, 
            detail="Час очікування вичерпано. Процес тренування триває у фоновому режимі."
        )
    except requests.exceptions.HTTPError:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Помилка ML-сервера: {response.text}"
        )

    # Save experiment to DB
    metrics = ml_data.get("metrics", {})
    exp_id = request.experiment_id or "default_exp"
    # Ensure unique experiment_id (append suffix if collision)
    base_exp_id = exp_id
    counter = 0
    while db.query(Experiment).filter(Experiment.experiment_id == exp_id).first():
        counter += 1
        exp_id = f"{base_exp_id}_{counter}"

    exp = Experiment(
        experiment_id=exp_id,
        user_id=current_user.id,
        dataset_id=active_ds.id,
        model_type=actual_model_type,
        model_file=ml_data.get("path"),
        accuracy=metrics.get("accuracy"),
        precision=metrics.get("precision"),
        recall=metrics.get("recall"),
        f1_score=metrics.get("f1_score"),
        train_size=metrics.get("train_size"),
        test_size=metrics.get("test_size"),
        training_time=str(metrics.get("training_time", "")),
        status="success",
        model_configs=json.dumps(request.models) if request.models else None,
    )
    db.add(exp)

    # Register model file
    model_file = ml_data.get("path")
    if model_file:
        filename = os.path.basename(model_file)
        if not db.query(ModelRecord).filter(ModelRecord.filename == filename).first():
            db.add(ModelRecord(
                experiment_id=exp_id,
                filename=filename,
                name=request.model_name or None,
                model_path=model_file,
                model_type=actual_model_type,
                accuracy=metrics.get("accuracy"),
            ))
    db.commit()

    return ml_data


# ── POST /evaluate ────────────────────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    max_samples: int = 50
    test_size: float = 0.2


@app.post("/evaluate")
def evaluate_llm(
    req: EvaluateRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Run Gemini zero-shot classification on the test split of the dataset.
    When IS_COLAB=true — proxies the entire evaluation to Colab (POST /run_evaluation).
    Otherwise — fetches dataset from local Flask and runs Gemini locally.
    """
    import time
    import re
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    )

    is_colab = os.getenv("IS_COLAB", "false").lower() in ("true", "1", "t")

    # ── Colab mode: proxy entire evaluation to Colab ──────────────────────────
    if is_colab:
        colab_base = os.getenv("COLAB_NGROK_URL", "").rstrip("/")
        if not colab_base:
            raise HTTPException(status_code=500, detail="COLAB_NGROK_URL not set")
        try:
            resp = requests.post(
                f"{colab_base}/run_evaluation",
                json={"max_samples": req.max_samples, "test_size": req.test_size},
                timeout=14400,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError:
            raise HTTPException(status_code=503, detail="Colab недоступний. Перевірте COLAB_NGROK_URL.")
        except requests.exceptions.HTTPError:
            raise HTTPException(status_code=resp.status_code, detail=f"Colab error: {resp.text}")

    # ── Local mode: fetch dataset from Flask, classify with local Gemini key ──
    try:
        ds_resp = requests.get(
            "http://127.0.0.1:5050/dataset/test",
            params={"size": req.test_size, "max_samples": req.max_samples},
            timeout=30,
        )
        ds_resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"ML-сервер недоступний: {e}")

    samples = ds_resp.json().get("samples", [])
    if not samples:
        raise HTTPException(status_code=400, detail="Dataset returned no samples")

    from api.llm_predictor import _get_client, _call_with_fallback

    client, types = _get_client()

    def classify_one(text: str) -> dict:
        result, _ = _call_with_fallback(
            client, types, f"Classify this text:\n\n{text[:1500]}",
            temperature=0, max_tokens=150,
        )
        return result

    results = []
    y_true, y_pred = [], []

    for i, sample in enumerate(samples):
        prediction = classify_one(sample["text"])
        pred_label = 1 if prediction["label"] == "FAKE" else 0
        true_label = int(sample["true_label"])
        results.append({
            "index": i + 1,
            "text": sample["text"][:200] + ("…" if len(sample["text"]) > 200 else ""),
            "true_label": true_label,
            "pred_label": pred_label,
            "pred_str": prediction["label"],
            "true_str": "FAKE" if true_label == 1 else "REAL",
            "confidence": round(prediction["confidence"], 3),
            "reason": prediction.get("reason", ""),
            "correct": pred_label == true_label,
        })
        y_true.append(true_label)
        y_pred.append(pred_label)
        time.sleep(1.1)

    acc  = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec  = float(recall_score(y_true, y_pred, zero_division=0))
    f1   = float(f1_score(y_true, y_pred, zero_division=0))
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "samples_evaluated": len(y_true),
        },
        "samples": results,
    }