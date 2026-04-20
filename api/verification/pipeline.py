# api/verification/pipeline.py
"""
Main verification pipeline — orchestrator.

Приймає текст (пост/твіт/твердження), проходить через 4 компоненти:
1. Claim extraction
2. Evidence retrieval (RSS + Social)
3. Stance detection
4. Verdict aggregation

Повертає VerificationResult.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from .models import (
    Claim,
    Evidence,
    EvidenceBundle,
    Verdict,
    VerificationResult,
)
from .claim_extractor import extract_claims
from .evidence_retriever import gather_evidence
from .stance_detector import detect_stance_batch
from .verdict_aggregator import aggregate_verdict, rule_based_verdict

logger = logging.getLogger(__name__)


async def verify_text(
    text: str,
    *,
    max_claims: int = 2,
    limit_rss: int = 5,
    limit_social: int = 10,
    sources: Optional[list[str]] = None,
    use_llm_verdict: bool = True,
) -> VerificationResult:
    """
    Full verification pipeline.

    Args:
        text: вхідний текст (пост/твіт/заява)
        max_claims: скільки claims обробляти (зазвичай достатньо 1-2)
        limit_rss: скільки RSS-статей шукати per claim
        limit_social: скільки social-постів per platform per claim
        sources: список джерел, або None → ["rss", "bluesky", "mastodon"]
        use_llm_verdict: LLM aggregation чи rule-based

    Returns:
        VerificationResult з усіма проміжними і фінальним результатом.
    """
    start = time.time()
    logger.info(f"verify_text: starting pipeline for text len={len(text)}")

    # Етап 1: Claim Extraction
    claims = await extract_claims(text, max_claims=max_claims)
    logger.info(f"extracted {len(claims)} claims")

    if not claims:
        # Спеціальний випадок — нічого не знайдено
        return VerificationResult(
            original_text=text,
            claims=[],
            evidence_bundles=[],
            verdicts=[],
            overall_verdict=_no_claim_verdict(),
            total_time_seconds=round(time.time() - start, 2),
        )

    # Етап 2: Evidence Retrieval (паралельно для всіх claims)
    evidence_bundles = await asyncio.gather(*[
        gather_evidence(claim, limit_rss=limit_rss, limit_social=limit_social, sources=sources)
        for claim in claims
    ])

    # Етап 3: Stance Detection (послідовно, щоб не перевищити LLM rate limit)
    for bundle in evidence_bundles:
        all_ev = bundle.all_evidence()
        if all_ev:
            logger.info(f"stance detection for claim '{bundle.claim.text[:60]}', {len(all_ev)} items")
            await detect_stance_batch(bundle.claim, all_ev, batch_size=5)

    # Етап 4: Verdict per claim
    verdicts = []
    for bundle in evidence_bundles:
        verdict = await aggregate_verdict(
            bundle.claim,
            bundle.all_evidence(),
            use_llm=use_llm_verdict,
        )
        verdicts.append(verdict)

    # Overall verdict якщо кілька claims
    overall = _compute_overall_verdict(verdicts)

    elapsed = time.time() - start
    logger.info(f"verify_text: done in {elapsed:.1f}s")

    return VerificationResult(
        original_text=text,
        claims=claims,
        evidence_bundles=evidence_bundles,
        verdicts=verdicts,
        overall_verdict=overall,
        total_time_seconds=round(elapsed, 2),
    )


def _no_claim_verdict() -> Verdict:
    """Verdict коли в тексті нема перевірюваних claim'ів."""
    from .models import EvidenceBreakdown
    return Verdict(
        label="UNCERTAIN",
        confidence=0.0,
        reasoning=(
            "Текст не містить перевірюваних фактичних тверджень "
            "(можливо, емоційна реакція, питання або особистий досвід)."
        ),
        breakdown=EvidenceBreakdown(),
        evidence_count=0,
        model_used="pipeline",
    )


def _compute_overall_verdict(verdicts: list[Verdict]) -> Optional[Verdict]:
    """
    Якщо один claim — повертаємо його verdict.
    Якщо кілька — обираємо найсильніший за confidence.
    (За бажанням можна агрегувати складніше, але MVP такий.)
    """
    if not verdicts:
        return None
    if len(verdicts) == 1:
        return verdicts[0]

    # Multi-claim: вибираємо найвпевненіший FAKE, інакше найвпевненіший взагалі
    fakes = [v for v in verdicts if v.label == "FAKE"]
    if fakes:
        return max(fakes, key=lambda v: v.confidence)

    return max(verdicts, key=lambda v: v.confidence)