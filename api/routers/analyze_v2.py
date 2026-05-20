"""Universal analysis gateway — orchestrates extraction + classification + verification.

POST /analyze/v2

Input modes:
  - "text":         raw text/article
  - "url":          Mastodon/Bluesky URL → fetch first
  - "claim_search": claim → search similar posts → batch analyze → aggregate spread

Note: окремий від існуючого `POST /analyze` (той — однокроковий classify-only,
викликається існуючим UI). Цей gateway довший і вмикає опціональні стадії.

Адаптовано до фактичних сигнатур:
  - `api.main.analyze_text` — синхронний (не async), порядок: (request, current_user, db)
  - `api.routers.sources.fetch_by_url` / `search_posts` — async, повертають dict/Pydantic
  - `api.fact_check.verify_post` — синхронний
  - `api.claim_extractor.extract_claims` — синхронний
"""
from __future__ import annotations

import logging
import time
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.auth import get_current_user
from api.claim_extractor import extract_claims as sync_extract_claims
from api.database import ModelRecord, User, get_db
from api.fact_check import verify_post

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analyze", tags=["analyze_v2"])


class AnalyzeV2Options(BaseModel):
    extract_claim: bool = Field(default=True, description="LLM extract claim+stance")
    classify: bool = Field(default=True, description="Run classifier")
    fact_check: bool = Field(default=False, description="Google Fact Check")
    search_sources: list[str] = Field(
        default_factory=lambda: ["mastodon", "bluesky"]
    )
    search_limit: int = Field(default=20, ge=1, le=50)
    classify_extracted: bool = Field(
        default=False,
        description="Якщо True — класифікувати extracted claim замість raw text",
    )
    explain: bool = Field(
        default=True,
        description=(
            "Запитувати local explanation моделі (NB log-odds / DistilBERT IG). "
            "Default True — UI завжди показує ExplanationPanel якщо backend "
            "повертає поле `explanation`."
        ),
    )


class AnalyzeV2Request(BaseModel):
    input_mode: Literal["text", "url", "claim_search"]
    input: str = Field(..., min_length=1, max_length=5000)
    model_id: Optional[int] = Field(
        default=None,
        description="ID моделі. None = auto-select",
    )
    options: AnalyzeV2Options = Field(default_factory=AnalyzeV2Options)


class AnalyzeV2Response(BaseModel):
    input_mode: str
    original_text: str
    fetched_post: Optional[dict] = None

    extraction: Optional[dict] = None
    extracted_claim: Optional[str] = None
    classification: Optional[dict] = None
    classified_text: Optional[str] = None
    classification_input: Optional[str] = None
    extraction_fallback: bool = False
    extraction_fallback_reason: Optional[str] = None
    model_used: Optional[dict] = None

    fact_check: Optional[dict] = None

    similar_posts: Optional[list[dict]] = None
    aggregated: Optional[dict] = None

    inference_context: Optional[dict] = None

    timing_ms: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


_PRIORITY_TYPES = ["llm", "distilbert", "deberta", "nb"]
_EXCLUDED_TYPES = ["gin", "sage", "gnn"]

_MIN_CLAIM_LEN = 10
_REFUSAL_TOKENS = (
    "cannot extract",
    "no claim found",
    "no verifiable claim",
    "не вдалося",
    "немає твердження",
)


def _validate_extracted_claim(claim: Optional[str]) -> tuple[bool, Optional[str]]:
    """Перевірити що claim придатний для класифікації.

    Returns (is_valid, reason_if_invalid).
    """
    if not claim or not isinstance(claim, str):
        return False, "empty"
    stripped = claim.strip()
    if len(stripped) < _MIN_CLAIM_LEN:
        return False, f"too_short ({len(stripped)} chars)"
    low = stripped.lower()
    for tok in _REFUSAL_TOKENS:
        if tok in low:
            return False, f"refusal ({tok!r})"
    return True, None


def _auto_select_model(db: Session) -> Optional[ModelRecord]:
    """Авто-вибір моделі для real-world аналізу.

    Пріоритет: active LLM → distilbert → deberta → nb. GNN-варіанти виключені
    (потребують tweet cascade structure, який ми не маємо для one-shot text).
    """
    for mtype in _PRIORITY_TYPES:
        model = (
            db.query(ModelRecord)
            .filter(
                ModelRecord.model_type == mtype,
                ModelRecord.is_active.is_(True),
            )
            .order_by(ModelRecord.f1_score.desc())
            .first()
        )
        if model:
            return model

    return (
        db.query(ModelRecord)
        .filter(
            ModelRecord.is_active.is_(True),
            ~ModelRecord.model_type.in_(_EXCLUDED_TYPES),
        )
        .order_by(ModelRecord.f1_score.desc())
        .first()
    )


def _classify_with_model(
    text: str,
    model_id: int,
    current_user: User,
    db: Session,
    *,
    explain: bool = False,
) -> dict:
    """Класифікувати текст через existing `/analyze` handler.

    `analyze_text` синхронний; викликаємо напряму як звичайну функцію.
    `explain` форвардиться → /analyze повертає `result["explanation"]`
    для NB і DistilBERT (через Colab /explain_distilbert).
    """
    from api.main import AnalyzeRequest, analyze_text

    req = AnalyzeRequest(text=text, model_id=model_id, explain=explain)
    return analyze_text(request=req, current_user=current_user, db=db)


def _aggregate_spread(posts_with_results: list[dict]) -> dict:
    """Aggregate класифікації N постів.

    Returns:
      - stance_distribution / classification_distribution
      - majority_verdict + consensus_strength
      - spread_warning якщо більшість поширює FAKE-claim як TRUE
    """
    if not posts_with_results:
        return {
            "total_posts": 0,
            "majority_verdict": "UNKNOWN",
            "consensus_strength": 0.0,
        }

    stance_counts = {"supports": 0, "refutes": 0, "neutral": 0}
    label_counts = {"FAKE": 0, "REAL": 0, "UNCERTAIN": 0}
    confidences: list[float] = []
    total_claims = 0

    for p in posts_with_results:
        extraction = p.get("extraction") or {}
        claims = extraction.get("claims") or []
        per_post = {"supports": 0, "refutes": 0, "neutral": 0}
        for claim in claims:
            s = claim.get("stance", "neutral")
            if s in per_post:
                per_post[s] += 1
                total_claims += 1
        if any(per_post.values()):
            post_stance = max(
                ("supports", "refutes", "neutral"),
                key=lambda s: (per_post[s], -("supports", "refutes", "neutral").index(s)),
            )
        else:
            post_stance = "neutral"
        stance_counts[post_stance] += 1

        cls = p.get("classification") or {}
        label = cls.get("label", "UNCERTAIN")
        if label in label_counts:
            label_counts[label] += 1
            if label != "UNCERTAIN":
                try:
                    confidences.append(float(cls.get("confidence", 0.5)))
                except (TypeError, ValueError):
                    pass

    total = len(posts_with_results)
    fake_pct = label_counts["FAKE"] / total
    real_pct = label_counts["REAL"] / total

    if fake_pct > 0.5:
        majority = "FAKE"
        consensus = fake_pct
    elif real_pct > 0.5:
        majority = "REAL"
        consensus = real_pct
    else:
        majority = "UNCERTAIN"
        consensus = max(fake_pct, real_pct)

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.5

    warning: Optional[str] = None
    if majority == "FAKE" and stance_counts["supports"] > stance_counts["refutes"]:
        ratio = stance_counts["supports"] / total
        if ratio > 0.5:
            warning = (
                f"{stance_counts['supports']} з {total} авторів "
                f"({ratio * 100:.0f}%) поширюють непідтверджену інформацію"
            )

    return {
        "total_posts": total,
        "total_claims": total_claims,
        "stance_distribution": stance_counts,
        "classification_distribution": label_counts,
        "majority_verdict": majority,
        "majority_confidence": round(avg_conf, 3),
        "consensus_strength": round(consensus, 3),
        "spread_warning": warning,
    }


async def _search_and_classify_similar(
    *,
    claim: str,
    sources: list[str],
    limit: int,
    model_id: int,
    current_user: User,
    db: Session,
) -> list[dict]:
    """Знайти схожі пости (через `/sources/search`) і збагатити кожен:
    extraction + classification.
    """
    from api.routers.sources import search_posts as search_endpoint

    valid_sources = [s for s in sources if s in {"bluesky", "mastodon"}]
    if not valid_sources:
        valid_sources = ["bluesky", "mastodon"]

    try:
        search_result = await search_endpoint(
            query=claim[:200],
            sources=",".join(valid_sources),
            limit=limit,
            current_user=current_user,
        )
    except HTTPException as e:
        logger.warning(f"Search failed: {e.detail}")
        return []
    except Exception as e:
        logger.error(f"Search unexpected error: {e}")
        return []

    posts = list(search_result.posts or [])
    if not posts:
        return []

    enriched: list[dict] = []
    for post in posts:
        post_dict = post if isinstance(post, dict) else {}
        post_text = post_dict.get("text", "") or post_dict.get("title", "")
        if not post_text or len(post_text.strip()) < 10:
            continue

        item: dict = {"post": post_dict}

        try:
            item["extraction"] = sync_extract_claims(post_text, use_llm=True)
        except Exception as e:
            logger.warning(f"Per-post extraction failed: {e}")
            item["extraction"] = {"claims": [], "method": "error"}

        try:
            item["classification"] = _classify_with_model(
                post_text, model_id, current_user, db
            )
        except HTTPException as e:
            item["classification"] = {
                "label": "UNCERTAIN",
                "confidence": 0.0,
                "error": str(e.detail),
            }
        except Exception as e:
            logger.warning(f"Per-post classification failed: {e}")
            item["classification"] = {
                "label": "UNCERTAIN",
                "confidence": 0.0,
                "error": str(e),
            }

        enriched.append(item)

    return enriched


@router.post("/v2", response_model=AnalyzeV2Response)
async def analyze_v2(
    req: AnalyzeV2Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Уніфікований analyzer для 3 режимів (text / url / claim_search)."""
    timings: dict[str, int] = {}
    warnings: list[str] = []

    t0 = time.time()
    if req.model_id is None:
        model = _auto_select_model(db)
        if not model:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Немає доступних моделей. Натренуйте модель або активуйте "
                    "LLM preset."
                ),
            )
        warnings.append(f"Авто-обрана модель: {model.name}")
    else:
        model = (
            db.query(ModelRecord).filter(ModelRecord.id == req.model_id).first()
        )
        if not model:
            raise HTTPException(
                status_code=404, detail=f"Модель з id={req.model_id} не знайдено"
            )
        if model.model_type in _EXCLUDED_TYPES and req.input_mode != "claim_search":
            warnings.append(
                "GIN/SAGE моделі потребують tweet cascade structure. "
                "Для one-shot text аналіз буде наближеним."
            )
    timings["model_select_ms"] = int((time.time() - t0) * 1000)

    original_text = req.input
    fetched_post: Optional[dict] = None

    if req.input_mode == "url":
        t1 = time.time()
        from api.routers.sources import FetchUrlRequest, fetch_by_url

        try:
            fetched_post = await fetch_by_url(
                FetchUrlRequest(url=req.input), current_user=current_user
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=502, detail=f"Не вдалося завантажити URL: {e}"
            )
        timings["fetch_url_ms"] = int((time.time() - t1) * 1000)

        if not isinstance(fetched_post, dict):
            raise HTTPException(
                status_code=502, detail="Fetcher повернув неочікуваний формат"
            )

        original_text = (
            fetched_post.get("text") or fetched_post.get("title") or ""
        ).strip()
        if not original_text:
            raise HTTPException(
                status_code=400, detail="Завантажений пост не містить тексту"
            )

    extraction: Optional[dict] = None
    if req.options.extract_claim:
        t2 = time.time()
        try:
            extraction = sync_extract_claims(original_text, use_llm=True)
        except Exception as e:
            logger.warning(f"Extraction failed: {e}")
            warnings.append(f"Extraction помилка: {e}")
        timings["extraction_ms"] = int((time.time() - t2) * 1000)

    inference_context: Optional[dict] = None
    inference_reqs_raw = getattr(model, "inference_requirements", None)
    if inference_reqs_raw:
        try:
            import json as _json
            inference_reqs = _json.loads(inference_reqs_raw)
        except (TypeError, ValueError):
            inference_reqs = None
    else:
        inference_reqs = None

    from api.inference_context import build_inference_context, needs_context
    if needs_context(inference_reqs):
        t_ctx = time.time()
        try:
            inference_context = await build_inference_context(
                text=original_text,
                inference_requirements=inference_reqs or {},
            )
            warnings.extend(
                (inference_context.get("metadata") or {}).get("warnings") or []
            )
        except Exception as e:
            logger.warning(f"inference_context build failed: {e}")
            warnings.append(f"inference_context_failed: {e}")
        timings["inference_context_ms"] = int((time.time() - t_ctx) * 1000)

    extracted_claim: Optional[str] = None
    if extraction and extraction.get("claims"):
        first = extraction["claims"][0]
        extracted_claim = (first.get("claim") or "").strip() or None

    text_to_classify = original_text
    extraction_fallback = False
    extraction_fallback_reason: Optional[str] = None

    if req.options.classify_extracted:
        if not extraction or not extraction.get("claims"):
            extraction_fallback = True
            extraction_fallback_reason = "no_extraction"
            warnings.append("Extraction повернула 0 claims — класифікуємо raw text")
        else:
            valid, reason = _validate_extracted_claim(extracted_claim)
            if valid:
                text_to_classify = extracted_claim
                warnings.append("Класифікуємо extracted claim замість raw text")
            else:
                extraction_fallback = True
                extraction_fallback_reason = reason
                warnings.append(
                    f"Extracted claim непридатний ({reason}) — класифікуємо raw text"
                )

    if req.options.classify and not (text_to_classify and text_to_classify.strip()):
        raise HTTPException(
            status_code=400,
            detail="Empty text after extraction — нічого класифікувати",
        )

    classification: Optional[dict] = None
    if req.options.classify:
        logger.info(
            "analyze/v2 → model=%s (id=%s) input_len=%d extraction_fallback=%s "
            "head=%r",
            model.model_type, model.id, len(text_to_classify or ""),
            extraction_fallback, (text_to_classify or "")[:120],
        )
        t3 = time.time()
        try:
            classification = _classify_with_model(
                text_to_classify, model.id, current_user, db,
                explain=req.options.explain,
            )
        except HTTPException as e:
            warnings.append(f"Classification помилка: {e.detail}")
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            warnings.append(f"Classification помилка: {e}")
        timings["classification_ms"] = int((time.time() - t3) * 1000)

    fact_check: Optional[dict] = None
    if req.options.fact_check and classification:
        t4 = time.time()
        try:
            fact_check = verify_post(
                original_text,
                model_label=classification.get("label"),
                model_confidence=classification.get("confidence"),
                language="en",
                use_llm_extraction=True,
            )
        except Exception as e:
            logger.warning(f"Fact-check failed: {e}")
            warnings.append(f"Fact-check помилка: {e}")
        timings["fact_check_ms"] = int((time.time() - t4) * 1000)

    similar_posts: Optional[list[dict]] = None
    aggregated: Optional[dict] = None

    if req.input_mode == "claim_search":
        t5 = time.time()
        try:
            similar_posts = await _search_and_classify_similar(
                claim=original_text,
                sources=req.options.search_sources,
                limit=req.options.search_limit,
                model_id=model.id,
                current_user=current_user,
                db=db,
            )
            aggregated = _aggregate_spread(similar_posts)
        except Exception as e:
            logger.warning(f"Claim search failed: {e}")
            warnings.append(f"Spread analysis помилка: {e}")
        timings["spread_analysis_ms"] = int((time.time() - t5) * 1000)

    return AnalyzeV2Response(
        input_mode=req.input_mode,
        original_text=original_text,
        fetched_post=fetched_post,
        extraction=extraction,
        extracted_claim=extracted_claim,
        classification=classification,
        classified_text=(
            text_to_classify if text_to_classify != original_text else None
        ),
        classification_input=text_to_classify if req.options.classify else None,
        extraction_fallback=extraction_fallback,
        extraction_fallback_reason=extraction_fallback_reason,
        model_used={
            "id": model.id,
            "name": model.name,
            "type": model.model_type,
            "f1_score": model.f1_score,
        },
        fact_check=fact_check,
        similar_posts=similar_posts,
        aggregated=aggregated,
        inference_context=inference_context,
        timing_ms=timings,
        warnings=warnings,
    )
