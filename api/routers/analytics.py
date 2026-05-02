"""
api/routers/analytics.py

Endpoint для отримання аналітики по датасету. Проксі на Colab.

Використання:
    GET /datasets/{id}/analytics?refresh=false
    
Повертає кешований результат з БД або запитує Colab.
"""
import json
import logging
import os

import requests
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.auth import get_current_user
from api.database import Dataset, User, get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["analytics"])


@router.get("/{dataset_id}/analytics")
def get_dataset_analytics(
    dataset_id: int,
    refresh: bool = False,
    sample_size: int = 5000,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Повертає аналітику по датасету.
    
    - refresh=false: спробувати кеш, якщо немає — обчислити
    - refresh=true: завжди перерахувати через Colab
    """
    dataset = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        Dataset.user_id == current_user.id,
    ).first()
    
    if not dataset:
        raise HTTPException(404, "Dataset not found")

    # Check cache
    if not refresh and hasattr(dataset, "analytics_cache") and dataset.analytics_cache:
        try:
            cached = json.loads(dataset.analytics_cache)
            logger.info(f"Returning cached analytics for dataset {dataset_id}")
            return cached
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Cached analytics corrupted for dataset {dataset_id}, recomputing")

    # Determine ML server URL
    is_colab = os.getenv("IS_COLAB", "false").lower() in ("true", "1", "t")
    if is_colab:
        base = os.getenv("COLAB_NGROK_URL", "").strip().rstrip("/")
        if not base:
            raise HTTPException(500, "COLAB_NGROK_URL не налаштований")
    else:
        base = "http://127.0.0.1:5050"

    target_url = f"{base}/analyze_dataset"
    payload = {
        "dataset_id": dataset.id,
        "dataset_name": dataset.name,
        "sample_size": sample_size,
    }

    logger.info(f"Requesting analytics from {target_url}")
    try:
        response = requests.post(target_url, json=payload, timeout=600)
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Colab unreachable: {e}")
        raise HTTPException(503, f"ML-сервер недоступний: {e}")
    except requests.exceptions.Timeout:
        raise HTTPException(504, "Timeout during analytics (>10 хв)")
    except requests.exceptions.HTTPError as e:
        logger.error(f"ML server error: {response.text[:300]}")
        raise HTTPException(response.status_code, f"ML server: {response.text[:300]}")

    # Save to cache
    try:
        if hasattr(dataset, "analytics_cache"):
            dataset.analytics_cache = json.dumps(result)
            db.commit()
            logger.info(f"Cached analytics for dataset {dataset_id}")
    except Exception as e:
        logger.warning(f"Failed to cache analytics: {e}")
        db.rollback()

    return result