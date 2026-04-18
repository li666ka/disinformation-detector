# api/routers/llm_presets.py
"""
Endpoints for LLM preset management.

POST   /llm-presets                  — create and save a preset
POST   /llm-presets/test             — try a config on one text WITHOUT saving
POST   /llm-presets/random-samples   — fetch random examples from TRAINING_DF
                                       (proxied to Colab if IS_COLAB=true)
GET    /llm-presets/defaults         — default system_prompt, cot_instruction, models list
"""
import json
import os
import time
import logging
import requests
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.database import get_db, ModelRecord, User
from api.auth import get_current_user
from api.schemas import (
    LLMPresetCreate,
    LLMPresetTestRequest,
    LLMPresetTestResponse,
    RandomSamplesRequest,
    RandomSamplesResponse,
    ModelRecordResponse,
    AVAILABLE_BASE_MODELS,
    AVAILABLE_MODES,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_COT_INSTRUCTION,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm-presets", tags=["llm-presets"])


# ── GET defaults ──────────────────────────────────────────────────────────

@router.get("/defaults")
def get_defaults(_: User = Depends(get_current_user)):
    """
    Return default prompts and available options for UI.
    Called by frontend when rendering the LLM preset form.
    """
    return {
        "base_models": AVAILABLE_BASE_MODELS,
        "modes": AVAILABLE_MODES,
        "default_system_prompt": DEFAULT_SYSTEM_PROMPT,
        "default_cot_instruction": DEFAULT_COT_INSTRUCTION,
        "default_bagging_n_calls": 3,
        "default_temperature": 0.0,
        "default_max_output_tokens": 200,
    }


# ── POST test (preview before save) ───────────────────────────────────────

@router.post("/test", response_model=LLMPresetTestResponse)
def test_preset_config(
    req: LLMPresetTestRequest,
    _: User = Depends(get_current_user),
):
    """
    Run a preset config on ONE sample text to preview the output.
    Does NOT save anything to DB. Used in frontend wizard before clicking Save.
    """
    from api.llm_predictor import predict_with_preset

    config = {
        "base_model": req.base_model,
        "mode": req.mode,
        "system_prompt": req.system_prompt,
        "temperature": req.temperature,
        "max_output_tokens": req.max_output_tokens,
        "few_shot_examples": [ex.model_dump() for ex in (req.few_shot_examples or [])],
        "cot_instruction": req.cot_instruction,
        "bagging_n_calls": req.bagging_n_calls or 3,
    }

    t0 = time.time()
    try:
        result = predict_with_preset(req.test_text, config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # SDK not installed
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("test_preset_config failed")
        raise HTTPException(status_code=503, detail=f"LLM call failed: {e}")
    elapsed = time.time() - t0

    return LLMPresetTestResponse(
        label=result["label"],
        confidence=result["confidence"],
        reason=result.get("reason", ""),
        base_model_used=result.get("base_model_used", "unknown"),
        elapsed_seconds=round(elapsed, 2),
    )


# ── POST create (save) ────────────────────────────────────────────────────

@router.post("", response_model=ModelRecordResponse, status_code=201)
def create_preset(
    req: LLMPresetCreate,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """Save LLM preset as a ModelRecord (model_type='llm')."""
    # Check unique name within user's presets (optional — we use global unique)
    existing = db.query(ModelRecord).filter(
        ModelRecord.name == req.name,
        ModelRecord.model_type == "llm",
    ).first()
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"LLM preset with name '{req.name}' already exists",
        )

    # Build config dict, trim unused fields
    config = {
        "base_model": req.base_model,
        "mode": req.mode,
        "system_prompt": req.system_prompt or DEFAULT_SYSTEM_PROMPT,
        "temperature": req.temperature,
        "max_output_tokens": req.max_output_tokens,
    }

    if req.mode == "few_shot":
        config["few_shot_examples"] = [ex.model_dump() for ex in (req.few_shot_examples or [])]
    elif req.mode == "cot":
        config["cot_instruction"] = req.cot_instruction or DEFAULT_COT_INSTRUCTION
    elif req.mode == "bagging":
        config["bagging_n_calls"] = req.bagging_n_calls or 3
        # Force min 0.7 for bagging diversity
        config["temperature"] = max(config["temperature"], 0.7)

    record = ModelRecord(
        name=req.name,
        model_type="llm",
        filename=None,
        model_path=None,
        llm_config=json.dumps(config, ensure_ascii=False),
        accuracy=None,
        is_active=False,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    logger.info(f"Created LLM preset: id={record.id}, name={req.name}, mode={req.mode}")
    return record


# ── POST random samples for few-shot picker ───────────────────────────────

@router.post("/random-samples", response_model=RandomSamplesResponse)
def get_random_samples(
    req: RandomSamplesRequest,
    _: User = Depends(get_current_user),
):
    """
    Get random FAKE/REAL examples from TRAINING_DF for few-shot.

    Two modes:
    1. IS_COLAB=true → proxy to Colab /dataset/samples endpoint
    2. IS_COLAB=false → fetch from local Flask ML server (if available)
    """
    is_colab = os.getenv("IS_COLAB", "false").lower() in ("true", "1", "t")

    if is_colab:
        colab_url = os.environ.get("COLAB_NGROK_URL", "").strip().rstrip("/")
        if not colab_url:
            raise HTTPException(status_code=500, detail="COLAB_NGROK_URL not set")
        try:
            resp = requests.post(
                f"{colab_url}/dataset/random_samples",
                json={"n_fake": req.n_fake, "n_real": req.n_real},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return RandomSamplesResponse(examples=data.get("examples", []))
        except requests.exceptions.ConnectionError:
            raise HTTPException(status_code=503, detail="Colab недоступний")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Colab error: {e}")

    # Local mode: try Flask ML server on port 5050
    try:
        resp = requests.post(
            "http://127.0.0.1:5050/dataset/random_samples",
            json={"n_fake": req.n_fake, "n_real": req.n_real},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return RandomSamplesResponse(examples=data.get("examples", []))
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Не вдалося отримати приклади з датасету: {e}. "
                   "Переконайтеся, що ML-сервер запущений або IS_COLAB=true.",
        )


# ── DELETE ────────────────────────────────────────────────────────────────

@router.delete("/{preset_id}")
def delete_preset(
    preset_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """Delete an LLM preset by ID."""
    record = db.query(ModelRecord).filter(
        ModelRecord.id == preset_id,
        ModelRecord.model_type == "llm",
    ).first()
    if not record:
        raise HTTPException(status_code=404, detail="LLM preset not found")
    db.delete(record)
    db.commit()
    return {"ok": True, "deleted_id": preset_id}