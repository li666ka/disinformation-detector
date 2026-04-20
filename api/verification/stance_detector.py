# api/verification/stance_detector.py
"""
Stance Detection — для кожного evidence визначити чи він підтримує/спростовує
/не пов'язаний з claim.

Використовує Gemini LLM. Працює batch-ом — обробляє до N evidence за 1 виклик,
щоб економити токени/час.

Вхід:
    claim: Claim
    evidence_list: list[Evidence]

Вихід:
    список Evidence з заповненими stance/stance_confidence/stance_reasoning
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from .models import Claim, Evidence

logger = logging.getLogger(__name__)


STANCE_DETECTION_PROMPT = """You are an expert fact-checking analyst. Your job is to determine if each piece of evidence supports, refutes, or is unrelated to a specific claim.

For each evidence item, classify its stance:
- "supports": evidence confirms or agrees with the claim
- "refutes": evidence contradicts or disputes the claim
- "unrelated": evidence mentions similar keywords but doesn't address the claim directly
- "unknown": evidence is ambiguous or too short to judge

For each piece of evidence, provide:
- stance: one of the 4 labels above
- confidence: 0.0 to 1.0 (how confident you are in the stance)
- reasoning: 1-2 short sentences explaining why

Respond ONLY with valid JSON array:
[
  {"stance": "...", "confidence": 0.9, "reasoning": "..."},
  ...
]

Preserve the order of input evidence items.
"""


async def detect_stance_batch(
    claim: Claim,
    evidence_list: list[Evidence],
    batch_size: int = 5,
) -> list[Evidence]:
    """
    Заповнити stance для кожного evidence у списку.

    Args:
        claim: Claim
        evidence_list: list[Evidence] (може бути empty)
        batch_size: скільки evidence оброблюємо за 1 LLM call

    Returns:
        Той самий список, з модифікованими Evidence (stance заповнений).
    """
    if not evidence_list:
        return evidence_list

    try:
        from api.llm_predictor import _get_client, _call_gemini_raw, DEFAULT_BASE_MODEL
    except ImportError as e:
        logger.error(f"Cannot import llm_predictor: {e}")
        for ev in evidence_list:
            ev.stance = "unknown"
            ev.stance_confidence = 0.0
            ev.stance_reasoning = "stance detection unavailable"
        return evidence_list

    try:
        client, types = _get_client()
    except Exception as e:
        logger.error(f"Failed to init Gemini client: {e}")
        for ev in evidence_list:
            ev.stance = "unknown"
        return evidence_list

    # Обробляємо батчами
    for i in range(0, len(evidence_list), batch_size):
        batch = evidence_list[i:i + batch_size]
        await _process_batch(client, types, claim, batch)

    return evidence_list


async def _process_batch(client, types, claim: Claim, batch: list[Evidence]):
    """Обробити один batch: викликати LLM, розпарсити, заповнити stance."""
    user_prompt = _build_batch_prompt(claim, batch)

    try:
        from api.llm_predictor import _call_gemini_raw, DEFAULT_BASE_MODEL
        raw_text, model_used = _call_gemini_raw(
            client, types,
            base_model=DEFAULT_BASE_MODEL,
            system_prompt=STANCE_DETECTION_PROMPT,
            user_prompt=user_prompt,
            temperature=0.0,
            max_output_tokens=min(1500, 200 * len(batch)),
        )
    except Exception as e:
        logger.warning(f"Stance LLM call failed: {e}")
        for ev in batch:
            ev.stance = "unknown"
            ev.stance_confidence = 0.0
            ev.stance_reasoning = f"LLM error: {str(e)[:100]}"
        return

    parsed = _parse_stance_response(raw_text, expected_count=len(batch))

    for ev, stance_data in zip(batch, parsed):
        if stance_data is None:
            ev.stance = "unknown"
            ev.stance_confidence = 0.0
            ev.stance_reasoning = "parse error"
            continue

        ev.stance = stance_data.get("stance", "unknown")
        ev.stance_confidence = float(stance_data.get("confidence", 0.5))
        ev.stance_reasoning = stance_data.get("reasoning", "")


def _build_batch_prompt(claim: Claim, batch: list[Evidence]) -> str:
    """
    Формує prompt з claim + нумерованим списком evidence.
    """
    parts = [f"CLAIM: {claim.text}\n"]
    parts.append(f"\nEvidence items ({len(batch)}):\n")

    for i, ev in enumerate(batch, 1):
        parts.append(f"\n--- Evidence {i} ---")
        parts.append(f"Source: {ev.source_name} ({ev.source_type})")
        if ev.title:
            parts.append(f"Title: {ev.title}")
        ev_text = ev.text[:400] if ev.text else ""  # обрізаємо щоб не перевантажити prompt
        parts.append(f"Text: {ev_text}")

    parts.append(f"\n\nClassify stance for each of the {len(batch)} evidence items above.")
    parts.append("Return JSON array with exactly " + str(len(batch)) + " items in the same order.")
    return "\n".join(parts)


def _parse_stance_response(response_text: str, expected_count: int) -> list[Optional[dict]]:
    """
    Парсить LLM відповідь у список stance dicts.
    Повертає список довжини expected_count, з None для items які не вдалося розпарсити.
    """
    if not response_text:
        return [None] * expected_count

    # Шукаємо JSON array
    match = re.search(r'\[.*\]', response_text, re.DOTALL)
    if not match:
        return [None] * expected_count

    try:
        data = json.loads(match.group())
        if not isinstance(data, list):
            return [None] * expected_count

        results: list[Optional[dict]] = []
        for item in data:
            if not isinstance(item, dict):
                results.append(None)
                continue
            stance = item.get("stance", "").lower()
            if stance not in ("supports", "refutes", "unrelated", "unknown"):
                stance = "unknown"
            try:
                conf = float(item.get("confidence", 0.5))
                conf = max(0.0, min(1.0, conf))
            except (ValueError, TypeError):
                conf = 0.5
            results.append({
                "stance": stance,
                "confidence": conf,
                "reasoning": str(item.get("reasoning", "")).strip()[:300],
            })

        # Доповнюємо до expected_count
        while len(results) < expected_count:
            results.append(None)

        return results[:expected_count]
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.debug(f"Stance JSON parse failed: {e}")
        return [None] * expected_count