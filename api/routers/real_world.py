"""API для аналізу окремих постів з Mastodon/Bluesky.

POST /real_world/analyze — fetch → LLM extraction → LLM classification (якщо
is_news_claim).
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Optional

import requests
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.auth import get_current_user
from api.database import User, get_db
from api.post_extractor import (
    ExtractedClaim,
    classify_claim,
    extract_post,
)
from api.social_fetchers import SocialPost, detect_platform, fetch_post

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/real_world", tags=["real_world"])


class AnalyzeRequest(BaseModel):
    url: str = Field(..., description="URL поста з Mastodon або Bluesky")
    extraction_model: str = Field(default="claude-haiku", description="Модель для extraction")
    classification_model: str = Field(default="claude-haiku", description="Модель для classification")


class AnalyzeResponse(BaseModel):
    platform: str
    post_url: str
    post_id: str
    author_handle: str
    author_display_name: Optional[str] = None
    content: str
    created_at: str
    language: Optional[str] = None

    engagement: dict
    media_count: int
    media_attachments: Optional[list[dict]] = None

    extraction: dict
    classification: Optional[dict] = None

    summary: str


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_post(
    req: AnalyzeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Проаналізувати один пост: fetch → extract → classify."""
    platform = detect_platform(req.url)
    if not platform:
        raise HTTPException(
            status_code=400,
            detail="Невідомий формат URL. Підтримуємо Mastodon та Bluesky.",
        )

    # ── Step 1: fetch ──
    try:
        post = fetch_post(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else 502
        raise HTTPException(
            status_code=502,
            detail=f"Платформа повернула HTTP {status}. Можливо пост приватний або видалений.",
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Не вдалося завантажити пост: {e}")

    logger.info(
        f"Fetched {post.platform} post {post.post_id} from {post.author_handle} "
        f"(content_len={len(post.content)})"
    )

    if not post.content.strip():
        raise HTTPException(
            status_code=400,
            detail="Пост не містить тексту (тільки медіа?). Аналіз неможливий.",
        )

    # ── Step 2: LLM extraction ──
    try:
        extraction = extract_post(post, model=req.extraction_model)
    except RuntimeError as e:
        logger.error(f"Extraction failed: {e}")
        raise HTTPException(status_code=503, detail=f"LLM extraction failed: {e}")

    # ── Step 3: classification (якщо є claim) ──
    classification: Optional[dict] = None
    if extraction.is_news_claim and extraction.claim_text:
        try:
            classification = classify_claim(
                extraction.claim_text, model=req.classification_model
            )
        except RuntimeError as e:
            logger.warning(f"Classification failed: {e}")
            classification = {"error": str(e), "verdict": "UNKNOWN"}

    summary = _build_summary(post, extraction, classification)

    return AnalyzeResponse(
        platform=post.platform,
        post_url=post.url,
        post_id=post.post_id,
        author_handle=post.author_handle,
        author_display_name=post.author_display_name,
        content=post.content,
        created_at=post.created_at,
        language=post.language,
        engagement={
            "replies": post.replies_count,
            "reposts": post.reposts_count,
            "likes": post.likes_count,
        },
        media_count=len(post.media_attachments or []),
        media_attachments=post.media_attachments,
        extraction=asdict(extraction),
        classification=classification,
        summary=summary,
    )


def _build_summary(
    post: SocialPost,
    extraction: ExtractedClaim,
    classification: Optional[dict],
) -> str:
    if not extraction.is_news_claim:
        return (
            "Пост не містить перевіряємого news claim. "
            f"{extraction.reasoning_for_is_news}"
        )
    if not classification:
        return (
            f"Знайдено claim: «{extraction.claim_text}», але classification недоступна."
        )

    verdict = classification.get("verdict", "UNKNOWN")
    conf = float(classification.get("confidence") or 0.0)
    verdict_uk = {
        "TRUE": "ймовірно ПРАВДА",
        "FALSE": "ймовірно ФЕЙК",
        "UNCERTAIN": "НЕВИЗНАЧЕНО",
        "UNKNOWN": "помилка classification",
    }.get(verdict, verdict)

    return (
        f"Claim: «{extraction.claim_text}» — {verdict_uk} "
        f"(впевненість {conf * 100:.0f}%). "
        f"Stance автора: {extraction.author_stance}."
    )
