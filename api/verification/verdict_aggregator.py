# api/verification/verdict_aggregator.py
"""
Verdict Aggregation — фінальний крок pipeline.

Вхід: Claim + list[Evidence] (з заповненими stance).
Вихід: Verdict (FAKE/REAL/UNCERTAIN + confidence + reasoning).

Два підходи реалізовані:
1. rule_based_verdict — швидкий, детермінований, weighted voting
2. llm_based_verdict — LLM синтезує reasoning на основі evidence

За замовчуванням aggregate_verdict() використовує LLM якщо є API key,
інакше rule-based fallback.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Optional

from .models import Claim, Evidence, EvidenceBreakdown, Verdict

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Rule-based aggregation
# ──────────────────────────────────────────────────────────────────────────

def rule_based_verdict(claim: Claim, evidence: list[Evidence]) -> Verdict:
    """
    Детермінований verdict на основі weighted voting.

    Логіка:
    - Для кожного evidence: якщо stance=supports/refutes, додаємо до weighted_X
      вагу = authority_weight * stance_confidence
    - Співвідношення weighted_refutes / (supports + refutes) визначає label:
      > 0.65 → FAKE
      < 0.35 → REAL (тобто більшість supports)
      in between → UNCERTAIN
    - Confidence пропорційний розмиру vote margin
    """
    start = time.time()
    breakdown = _compute_breakdown(evidence)

    total_weighted = breakdown.weighted_supports + breakdown.weighted_refutes

    if total_weighted < 0.1:
        # Немає авторитетного evidence
        return Verdict(
            label="UNCERTAIN",
            confidence=0.0,
            reasoning="Недостатньо авторитетних evidence для висновку.",
            breakdown=breakdown,
            evidence_count=len(evidence),
            highest_authority_evidence=_find_highest_authority(evidence),
            model_used="rule-based",
            verdict_time_seconds=round(time.time() - start, 2),
        )

    refute_ratio = breakdown.weighted_refutes / total_weighted
    margin = abs(refute_ratio - 0.5) * 2  # 0.0 at 50/50, 1.0 at 100/0

    if refute_ratio >= 0.65:
        label = "FAKE"
        confidence = min(0.95, 0.5 + margin * 0.5)
        reasoning = _build_rule_reasoning(breakdown, "refutes")
    elif refute_ratio <= 0.35:
        label = "REAL"
        confidence = min(0.95, 0.5 + margin * 0.5)
        reasoning = _build_rule_reasoning(breakdown, "supports")
    else:
        label = "UNCERTAIN"
        confidence = max(0.0, 0.4 - margin * 0.4)
        reasoning = f"Evidence показує суперечливі сигнали: {breakdown.supports} підтверджують, {breakdown.refutes} спростовують claim."

    return Verdict(
        label=label,
        confidence=round(confidence, 3),
        reasoning=reasoning,
        breakdown=breakdown,
        evidence_count=len(evidence),
        highest_authority_evidence=_find_highest_authority(evidence),
        model_used="rule-based",
        verdict_time_seconds=round(time.time() - start, 2),
    )


def _compute_breakdown(evidence: list[Evidence]) -> EvidenceBreakdown:
    """Підрахувати evidence по категоріях."""
    br = EvidenceBreakdown()

    for ev in evidence:
        stance = ev.stance or "unknown"
        weight = (ev.authority_weight or 0.5) * (ev.stance_confidence or 0.5)

        if stance == "supports":
            br.supports += 1
            br.weighted_supports += weight
        elif stance == "refutes":
            br.refutes += 1
            br.weighted_refutes += weight
        elif stance == "unrelated":
            br.unrelated += 1
        else:
            br.unknown += 1

    br.weighted_supports = round(br.weighted_supports, 3)
    br.weighted_refutes = round(br.weighted_refutes, 3)
    return br


def _find_highest_authority(evidence: list[Evidence]) -> Optional[Evidence]:
    """Знайти item з найбільшою authority weight серед stance=supports|refutes."""
    relevant = [e for e in evidence if e.stance in ("supports", "refutes")]
    if not relevant:
        return None
    return max(relevant, key=lambda e: (e.authority_weight or 0) * (e.stance_confidence or 0))


def _build_rule_reasoning(breakdown: EvidenceBreakdown, direction: str) -> str:
    """Згенерувати reasoning для rule-based verdict."""
    if direction == "refutes":
        return (
            f"{breakdown.refutes} з {breakdown.refutes + breakdown.supports} релевантних джерел "
            f"спростовують це твердження (зважений рахунок: {breakdown.weighted_refutes:.2f} "
            f"проти {breakdown.weighted_supports:.2f})."
        )
    else:
        return (
            f"{breakdown.supports} з {breakdown.supports + breakdown.refutes} релевантних джерел "
            f"підтверджують це твердження (зважений рахунок: {breakdown.weighted_supports:.2f} "
            f"проти {breakdown.weighted_refutes:.2f})."
        )


# ──────────────────────────────────────────────────────────────────────────
# LLM-based aggregation
# ──────────────────────────────────────────────────────────────────────────

VERDICT_PROMPT = """You are a senior fact-checker synthesizing evidence to reach a verdict on a factual claim.

You will receive:
1. A claim to verify
2. Evidence items, each with its stance (supports/refutes/unrelated) and source authority

Your task: reach a final verdict with confidence and reasoning.

Guidelines:
- Authoritative sources (major news, fact-checkers) outweigh individual social posts
- Multiple independent sources agreeing strengthens the verdict
- If evidence is contradictory or insufficient, say "UNCERTAIN"
- Reasoning should cite specific evidence sources (e.g., "BBC reports that...")
- Keep reasoning to 2-4 sentences, readable by a non-expert

Respond ONLY with valid JSON:
{
  "label": "FAKE" | "REAL" | "UNCERTAIN",
  "confidence": 0.0 to 1.0,
  "reasoning": "..."
}
"""


async def llm_based_verdict(
    claim: Claim,
    evidence: list[Evidence],
    rule_fallback: Optional[Verdict] = None,
) -> Verdict:
    """
    LLM синтезує verdict на основі evidence.

    Args:
        claim: Claim
        evidence: list[Evidence] з заповненими stance
        rule_fallback: Verdict який повертаємо у разі помилки LLM

    Returns:
        Verdict
    """
    start = time.time()
    breakdown = _compute_breakdown(evidence)

    # Якщо evidence undersized — просто rule-based
    if len([e for e in evidence if e.stance in ("supports", "refutes")]) == 0:
        return rule_fallback or rule_based_verdict(claim, evidence)

    try:
        from api.llm_predictor import _call_claude_cli, ClaudeCLIError, DEFAULT_BASE_MODEL
    except Exception as e:
        logger.warning(f"LLM unavailable for verdict, falling back to rules: {e}")
        return rule_fallback or rule_based_verdict(claim, evidence)

    prompt = _build_verdict_prompt(claim, evidence)
    full_prompt = f"{VERDICT_PROMPT}\n\n{prompt}"
    model_used = DEFAULT_BASE_MODEL

    try:
        raw_text = _call_claude_cli(full_prompt, model=DEFAULT_BASE_MODEL)
    except ClaudeCLIError as e:
        logger.warning(f"Verdict Claude CLI call failed: {e}")
        return rule_fallback or rule_based_verdict(claim, evidence)
    except Exception as e:
        logger.warning(f"Verdict unexpected error: {e}")
        return rule_fallback or rule_based_verdict(claim, evidence)

    # Parse the raw JSON response {label, confidence, reasoning}
    label_raw = "UNCERTAIN"
    confidence = 0.5
    reasoning = "Висновок на основі analyzed evidence."
    if raw_text:
        import re as _re
        match = _re.search(r'\{.*\}', raw_text, _re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                label_raw = data.get("label", "UNCERTAIN").upper()
                if label_raw not in ("FAKE", "REAL", "UNCERTAIN"):
                    label_raw = "UNCERTAIN"
                confidence = float(data.get("confidence", 0.5))
                reasoning = data.get("reasoning", "") or data.get("reason", "") or reasoning
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

    return Verdict(
        label=label_raw,  # type: ignore
        confidence=round(confidence, 3),
        reasoning=reasoning,
        breakdown=breakdown,
        evidence_count=len(evidence),
        highest_authority_evidence=_find_highest_authority(evidence),
        model_used=f"llm:{model_used}",
        verdict_time_seconds=round(time.time() - start, 2),
    )


def _build_verdict_prompt(claim: Claim, evidence: list[Evidence]) -> str:
    """Формує prompt з claim + summary evidence."""
    parts = [f"CLAIM: {claim.text}\n"]

    # Groupped by stance
    supports = [e for e in evidence if e.stance == "supports"]
    refutes = [e for e in evidence if e.stance == "refutes"]
    unrelated = [e for e in evidence if e.stance == "unrelated"]

    if supports:
        parts.append(f"\nEVIDENCE SUPPORTING THE CLAIM ({len(supports)} items):")
        for ev in supports[:5]:
            parts.append(_format_evidence_for_prompt(ev))

    if refutes:
        parts.append(f"\nEVIDENCE REFUTING THE CLAIM ({len(refutes)} items):")
        for ev in refutes[:5]:
            parts.append(_format_evidence_for_prompt(ev))

    if unrelated:
        parts.append(f"\n(Also found {len(unrelated)} unrelated items — not included)")

    parts.append("\nBased on this evidence, reach a verdict. Respond with JSON only.")
    return "\n".join(parts)


def _format_evidence_for_prompt(ev: Evidence) -> str:
    """Компактне форматування Evidence для prompt."""
    authority = ev.authority_weight or 0.5
    auth_label = "high" if authority >= 0.8 else ("medium" if authority >= 0.5 else "low")
    title_or_start = ev.title or ev.text[:100]
    return f"  - [{ev.source_name}, authority: {auth_label}] {title_or_start}"


# ──────────────────────────────────────────────────────────────────────────
# Main entry — auto-select aggregation strategy
# ──────────────────────────────────────────────────────────────────────────

async def aggregate_verdict(
    claim: Claim,
    evidence: list[Evidence],
    *,
    use_llm: bool = True,
) -> Verdict:
    """
    Головна функція aggregation. Автоматично вибирає стратегію.

    Args:
        claim: Claim
        evidence: list[Evidence] з заповненими stance
        use_llm: якщо True → пробує LLM, fallback на rules; False → тільки rules

    Returns:
        Verdict
    """
    rule_verdict = rule_based_verdict(claim, evidence)

    if not use_llm:
        return rule_verdict

    # Для consistency — використовуємо LLM тільки якщо є достатньо evidence
    relevant_count = rule_verdict.breakdown.supports + rule_verdict.breakdown.refutes
    if relevant_count == 0:
        return rule_verdict

    return await llm_based_verdict(claim, evidence, rule_fallback=rule_verdict)