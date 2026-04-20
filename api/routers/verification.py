# api/routers/verification.py
"""
Router для multi-hop claim verification.

Endpoints:
- POST /verification/verify — full pipeline на тексті
- POST /verification/extract-claims — тільки claim extraction (для debug)
- POST /verification/gather-evidence — тільки evidence retrieval (для debug)
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.auth import get_current_user
from api.database import User
from api.verification import (
    Claim,
    verify_text,
    extract_claims,
    gather_evidence,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/verification", tags=["verification"])


# ──────────────────────────────────────────────────────────────────────────
# Request/Response schemas
# ──────────────────────────────────────────────────────────────────────────

class VerifyRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=5000)
    max_claims: int = Field(default=2, ge=1, le=5)
    limit_rss: int = Field(default=5, ge=1, le=15)
    limit_social: int = Field(default=10, ge=1, le=30)
    sources: Optional[list[str]] = Field(default=None)
    use_llm_verdict: bool = Field(default=True)


class VerifyResponse(BaseModel):
    original_text: str
    claims: list[dict]
    evidence_bundles: list[dict]
    verdicts: list[dict]
    overall_verdict: Optional[dict] = None
    total_time_seconds: float


class ExtractClaimsRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=5000)
    max_claims: int = Field(default=3, ge=1, le=10)


class ExtractClaimsResponse(BaseModel):
    claims: list[dict]
    count: int


class GatherEvidenceRequest(BaseModel):
    claim_text: str = Field(..., min_length=5, max_length=500)
    entities: list[str] = Field(default_factory=list)
    limit_rss: int = Field(default=5, ge=1, le=15)
    limit_social: int = Field(default=10, ge=1, le=30)
    sources: Optional[list[str]] = Field(default=None)


class GatherEvidenceResponse(BaseModel):
    claim: dict
    evidence: dict  # EvidenceBundle dict


# ──────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────

@router.post("/verify", response_model=VerifyResponse)
async def verify_endpoint(
    req: VerifyRequest,
    _: User = Depends(get_current_user),
):
    """
    Full multi-hop verification pipeline.

    Послідовність:
    1. Витягнути claims з тексту через LLM
    2. Для кожного claim — зібрати evidence (RSS + соц-мережі)
    3. Для кожного evidence — визначити stance
    4. Агрегувати у verdict

    Очікуваний час виконання: 10-30 секунд (залежить від кількості evidence).
    """
    valid_sources = {"rss", "bluesky", "mastodon"}
    if req.sources:
        invalid = [s for s in req.sources if s not in valid_sources]
        if invalid:
            raise HTTPException(400, f"Invalid sources: {invalid}. Allowed: {sorted(valid_sources)}")

    try:
        result = await verify_text(
            req.text,
            max_claims=req.max_claims,
            limit_rss=req.limit_rss,
            limit_social=req.limit_social,
            sources=req.sources,
            use_llm_verdict=req.use_llm_verdict,
        )
    except Exception as e:
        logger.exception("verification pipeline failed")
        raise HTTPException(500, f"Verification failed: {e}")

    return VerifyResponse(**result.to_dict())


@router.post("/extract-claims", response_model=ExtractClaimsResponse)
async def extract_claims_endpoint(
    req: ExtractClaimsRequest,
    _: User = Depends(get_current_user),
):
    """Debug endpoint: тільки claim extraction."""
    try:
        claims = await extract_claims(req.text, max_claims=req.max_claims)
    except Exception as e:
        logger.exception("extract_claims failed")
        raise HTTPException(500, f"Claim extraction failed: {e}")

    return ExtractClaimsResponse(
        claims=[c.to_dict() for c in claims],
        count=len(claims),
    )


@router.post("/gather-evidence", response_model=GatherEvidenceResponse)
async def gather_evidence_endpoint(
    req: GatherEvidenceRequest,
    _: User = Depends(get_current_user),
):
    """Debug endpoint: evidence retrieval для конкретного claim."""
    valid_sources = {"rss", "bluesky", "mastodon"}
    if req.sources:
        invalid = [s for s in req.sources if s not in valid_sources]
        if invalid:
            raise HTTPException(400, f"Invalid sources: {invalid}")

    claim = Claim(
        text=req.claim_text,
        entities=req.entities,
    )

    try:
        bundle = await gather_evidence(
            claim,
            limit_rss=req.limit_rss,
            limit_social=req.limit_social,
            sources=req.sources,
        )
    except Exception as e:
        logger.exception("gather_evidence failed")
        raise HTTPException(500, f"Evidence retrieval failed: {e}")

    return GatherEvidenceResponse(
        claim=claim.to_dict(),
        evidence=bundle.to_dict(),
    )