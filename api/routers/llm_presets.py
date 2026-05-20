"""
Endpoints for LLM preset management.

POST   /llm-presets                  — create and save a preset
POST   /llm-presets/test             — try a config on one text WITHOUT saving
POST   /llm-presets/random-samples   — fetch random examples from local train split
GET    /llm-presets/defaults         — default system_prompt, cot_instruction, models list
"""
import json
import random
import time
import logging
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.database import get_db, ModelRecord, User, Dataset
from api.auth import get_current_user
from api.schemas import (
    LLMPresetCreate,
    LLMPresetUpdate,
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


@router.post("", response_model=ModelRecordResponse, status_code=201)
def create_preset(
    req: LLMPresetCreate,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """Save LLM preset as a ModelRecord (model_type='llm')."""
    existing = db.query(ModelRecord).filter(
        ModelRecord.name == req.name,
        ModelRecord.model_type == "llm",
    ).first()
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"LLM preset with name '{req.name}' already exists",
        )

    config = {
        "base_model": req.base_model,
        "mode": req.mode,
        "system_prompt": req.system_prompt or DEFAULT_SYSTEM_PROMPT,
        "temperature": req.temperature,
        "max_output_tokens": req.max_output_tokens,
        "include_social_context": req.include_social_context,
    }

    if req.mode == "few_shot":
        config["few_shot_examples"] = [ex.model_dump() for ex in (req.few_shot_examples or [])]
    elif req.mode == "cot":
        config["cot_instruction"] = req.cot_instruction or DEFAULT_COT_INSTRUCTION
    elif req.mode == "bagging":
        config["bagging_n_calls"] = req.bagging_n_calls or 3
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


@router.post("/random-samples", response_model=RandomSamplesResponse)
def get_random_samples(
    req: RandomSamplesRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get stratified random FAKE/REAL examples from the user's active dataset
    train split: uploaded_datasets/user_<id>/dataset_<id>/<splits_subdir>/train.csv
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

    ds_path = Path("uploaded_datasets") / f"user_{current_user.id}" / f"dataset_{active_ds.id}"
    news_path = ds_path / "news.csv"
    train_path = ds_path / req.splits_subdir / "train.csv"

    if not news_path.exists():
        raise HTTPException(status_code=404, detail=f"news.csv не знайдено: {news_path}")
    if not train_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"train.csv не знайдено для {req.splits_subdir}: {train_path}",
        )

    try:
        news_df = pd.read_csv(news_path, low_memory=False)
        train_split = pd.read_csv(train_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Помилка читання даних: {e}")

    train_aids = train_split["article_id"].astype(str).tolist()
    news_df["article_id"] = news_df["article_id"].astype(str)
    train_df = news_df[news_df["article_id"].isin(train_aids)].copy()

    if train_df.empty:
        raise HTTPException(
            status_code=500,
            detail="train.csv не співпадає з news.csv (всі IDs відсутні)",
        )

    fake_pool = train_df[train_df["article_label"].astype(str).str.upper() == "FAKE"]
    real_pool = train_df[train_df["article_label"].astype(str).str.upper() == "REAL"]

    if len(fake_pool) < req.n_fake:
        raise HTTPException(
            status_code=400,
            detail=f"Недостатньо FAKE прикладів у train: {len(fake_pool)} < {req.n_fake}",
        )
    if len(real_pool) < req.n_real:
        raise HTTPException(
            status_code=400,
            detail=f"Недостатньо REAL прикладів у train: {len(real_pool)} < {req.n_real}",
        )

    fake_samples = fake_pool.sample(n=req.n_fake) if req.n_fake > 0 else fake_pool.iloc[0:0]
    real_samples = real_pool.sample(n=req.n_real) if req.n_real > 0 else real_pool.iloc[0:0]

    examples = []
    for _, row in fake_samples.iterrows():
        text = (
            str(row.get("article_title", "") or "") + ". " +
            str(row.get("article_text", "") or "")
        ).strip()
        if len(text) > 500:
            text = text[:497] + "..."
        examples.append({"text": text, "label": "FAKE"})

    for _, row in real_samples.iterrows():
        text = (
            str(row.get("article_title", "") or "") + ". " +
            str(row.get("article_text", "") or "")
        ).strip()
        if len(text) > 500:
            text = text[:497] + "..."
        examples.append({"text": text, "label": "REAL"})

    random.shuffle(examples)

    return RandomSamplesResponse(examples=examples)


@router.patch("/{preset_id}", response_model=ModelRecordResponse)
def update_preset(
    preset_id: int,
    req: LLMPresetUpdate,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """Rename an LLM preset."""
    record = db.query(ModelRecord).filter(
        ModelRecord.id == preset_id,
        ModelRecord.model_type == "llm",
    ).first()
    if not record:
        raise HTTPException(status_code=404, detail="LLM preset not found")

    new_name = req.name.strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="Name cannot be empty")

    if new_name != record.name:
        clash = db.query(ModelRecord).filter(
            ModelRecord.name == new_name,
            ModelRecord.model_type == "llm",
            ModelRecord.id != preset_id,
        ).first()
        if clash:
            raise HTTPException(
                status_code=400,
                detail=f"LLM preset with name '{new_name}' already exists",
            )
        record.name = new_name
        db.commit()
        db.refresh(record)
        logger.info(f"Renamed LLM preset id={preset_id} -> '{new_name}'")
    return record


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