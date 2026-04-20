# api/verification/claim_extractor.py
"""
Claim Extraction — витягує перевірювані твердження з тексту.

Використовує Gemini LLM, через той самий google-genai SDK що вже
інтегрований у api/llm_predictor.py.

Вхід:
    text: довільний текст (пост, твіт, стаття)

Вихід:
    list[Claim] — 0+ claims. Для emotional reaction без фактів повертає [].
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from .models import Claim

logger = logging.getLogger(__name__)

CLAIM_EXTRACTION_PROMPT = """You are an expert in fact-checking and media analysis. Your task is to extract verifiable factual claims from the given text.

A "claim" is a statement that asserts a fact about the world and can be verified (or refuted) using evidence. It is NOT:
- An opinion ("I think X is bad")
- An emotional reaction ("OMG!")
- A question ("Is X true?")
- A personal experience ("I saw X")
- A speculation ("Maybe X")

For each claim, extract:
- text: the claim as a clear standalone sentence (rewrite for clarity if needed)
- claim_type: "statement" | "causal" | "statistical" | "quote"
  - statement: plain factual assertion ("X happened")
  - causal: cause-effect ("X causes Y")
  - statistical: numeric claim ("90% of X")
  - quote: attributed quote ("Person X said Y")
- entities: 2-5 key named entities (proper nouns, products, places, people)
- verifiable: true if fact-checkable, false if too vague

If the text contains NO verifiable claims, return an empty list.

Respond ONLY with valid JSON:
{
  "claims": [
    {"text": "...", "claim_type": "...", "entities": [...], "verifiable": true}
  ]
}
"""


async def extract_claims(text: str, max_claims: int = 3) -> list[Claim]:
    """
    Витягнути claims з тексту через Gemini.

    Args:
        text: вхідний текст
        max_claims: максимум claims (typically 1-3 достатньо, обрізаємо)

    Returns:
        Список Claim об'єктів. Порожній якщо нічого не знайдено.
    """
    if not text or len(text.strip()) < 10:
        return []

    text = text.strip()[:2000]  # обрізаємо до розумного розміру для LLM

    try:
        from api.llm_predictor import _get_client, _call_gemini_raw, DEFAULT_BASE_MODEL
    except ImportError as e:
        logger.error(f"Cannot import llm_predictor: {e}")
        return []

    try:
        client, types = _get_client()
    except Exception as e:
        logger.error(f"Failed to init Gemini client: {e}")
        return []

    user_prompt = f"Extract verifiable claims from this text:\n\n{text}"

    try:
        raw_text, model_used = _call_gemini_raw(
            client, types,
            base_model=DEFAULT_BASE_MODEL,
            system_prompt=CLAIM_EXTRACTION_PROMPT,
            user_prompt=user_prompt,
            temperature=0.0,
            max_output_tokens=500,
        )
    except Exception as e:
        logger.error(f"Gemini call failed for claim extraction: {e}")
        return []

    if not raw_text:
        logger.warning("Empty response from Gemini for claim extraction")
        return []

    parsed_claims = _parse_claims_json(raw_text, original_text=text)

    if not parsed_claims:
        # Model might have put JSON in alternative format — try regex fallback
        parsed_claims = _parse_claims_fallback(raw_text, original_text=text)

    return parsed_claims[:max_claims]


def _parse_claims_json(response_text: str, original_text: str) -> list[Claim]:
    """Парсить відповідь LLM у форматі {"claims": [...]}"""
    if not response_text:
        return []

    # Шукаємо JSON у відповіді
    match = re.search(r'\{.*"claims".*\}', response_text, re.DOTALL)
    if not match:
        return []

    try:
        data = json.loads(match.group())
        raw_claims = data.get("claims", [])
        if not isinstance(raw_claims, list):
            return []

        claims = []
        for rc in raw_claims:
            if not isinstance(rc, dict):
                continue
            claim_text = rc.get("text", "").strip()
            if not claim_text or len(claim_text) < 10:
                continue
            claims.append(Claim(
                text=claim_text,
                claim_type=rc.get("claim_type", "statement"),
                entities=list(rc.get("entities", [])) if isinstance(rc.get("entities"), list) else [],
                verifiable=bool(rc.get("verifiable", True)),
                original_text=original_text,
            ))
        return claims
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.debug(f"JSON parse failed: {e}")
        return []


def _parse_claims_fallback(response_text: str, original_text: str) -> list[Claim]:
    """
    Fallback: якщо LLM повернув не-JSON, але текст явно містить claims у bullets
    або нумерованих списках, спробувати витягти.
    """
    if not response_text:
        return []

    # Шукаємо bullet/numbered list
    lines = response_text.split("\n")
    claims = []
    for line in lines:
        line = line.strip()
        # Прибираємо bullet markers
        line = re.sub(r'^[\d\.\-\*•]+\s*', '', line)
        line = line.strip("\"' ")
        if len(line) < 15 or len(line) > 300:
            continue
        # Простий heuristic: claim має містити дієслово
        if not re.search(r'\b(is|was|are|were|has|have|had|causes?|makes?|says?|claims?|shows?)\b', line, re.IGNORECASE):
            continue
        claims.append(Claim(
            text=line,
            claim_type="statement",
            entities=[],
            verifiable=True,
            original_text=original_text,
        ))
        if len(claims) >= 3:
            break

    return claims


# ──────────────────────────────────────────────────────────────────────────
# Synchronous helper for non-async contexts (e.g., scripts)
# ──────────────────────────────────────────────────────────────────────────

def extract_claims_sync(text: str, max_claims: int = 3) -> list[Claim]:
    """Synchronous wrapper для скриптів. Використовуйте extract_claims у async."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError("Use extract_claims() directly in async context")
    except RuntimeError:
        pass
    return asyncio.run(extract_claims(text, max_claims))