# api/verification/claim_extractor.py
"""
Claim Extraction — витягує перевірювані твердження з тексту.

Викликає Claude напряму через `claude -p` subprocess
(api.llm_predictor._call_claude_cli) — без API token, через Max plan OAuth.

Вхід:
    text: довільний текст (пост, твіт, стаття)

Вихід:
    list[Claim] — 0+ claims. Для emotional reaction без фактів повертає [].

Per-text processing — НЕ batch, оскільки stance detection потребує
аналізу кожного тексту окремо.
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
- text: the claim as a clear standalone sentence (rewrite for clarity if needed, resolve pronouns)
- claim_type: "statement" | "causal" | "statistical" | "quote"
  - statement: plain factual assertion ("X happened")
  - causal: cause-effect ("X causes Y")
  - statistical: numeric claim ("90% of X")
  - quote: attributed quote ("Person X said Y")
- entities: 2-5 key named entities (proper nouns, products, places, people)
- verifiable: true if fact-checkable, false if too vague
- stance: "supports" | "refutes" | "neutral"
  - supports: author asserts the claim is TRUE
  - refutes: author asserts the claim is FALSE (e.g., "X does NOT cause Y")
  - neutral: author just mentions/quotes without taking position

If the text contains NO verifiable claims, return an empty list.

Respond ONLY with valid JSON:
{
  "claims": [
    {"text": "...", "claim_type": "...", "entities": [...], "verifiable": true, "stance": "supports"}
  ]
}
"""


async def extract_claims(text: str, max_claims: int = 3) -> list[Claim]:
    """
    Витягнути claims з тексту через Claude.

    Args:
        text: вхідний текст
        max_claims: максимум claims (typically 1-3 достатньо, обрізаємо)

    Returns:
        Список Claim об'єктів. Порожній якщо нічого не знайдено.
    """
    if not text or len(text.strip()) < 10:
        return []

    text = text.strip()[:2000]

    try:
        from api.llm_predictor import _call_claude_cli, ClaudeCLIError, DEFAULT_BASE_MODEL
    except ImportError as e:
        logger.error(f"Cannot import llm_predictor: {e}")
        return []

    full_prompt = (
        f"{CLAIM_EXTRACTION_PROMPT}\n\n"
        f"Extract verifiable claims from this text:\n\n{text}"
    )

    try:
        raw_text = _call_claude_cli(full_prompt, model=DEFAULT_BASE_MODEL)
    except ClaudeCLIError as e:
        logger.warning(f"Claude CLI failed for claim extraction: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in claim extraction: {e}")
        return []

    if not raw_text:
        logger.warning("Empty response from Claude for claim extraction")
        return []

    parsed_claims = _parse_claims_json(raw_text, original_text=text)

    if not parsed_claims:
        # Fallback regex if model returned non-JSON
        parsed_claims = _parse_claims_fallback(raw_text, original_text=text)

    return parsed_claims[:max_claims]


def _parse_claims_json(response_text: str, original_text: str) -> list[Claim]:
    """Парсить відповідь LLM у форматі {"claims": [...]}"""
    if not response_text:
        return []

    data = _extract_json_object(response_text)
    if data is None:
        return []

    try:
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

            # Build Claim — перевірити чи модель Claim підтримує stance
            claim_kwargs = {
                "text": claim_text,
                "claim_type": rc.get("claim_type", "statement"),
                "entities": list(rc.get("entities", [])) if isinstance(rc.get("entities"), list) else [],
                "verifiable": bool(rc.get("verifiable", True)),
                "original_text": original_text,
            }

            # Optional stance — якщо модель Claim має це поле
            stance = rc.get("stance", "neutral")
            if stance in ("supports", "refutes", "neutral"):
                # Try to set it; ignore якщо Claim не має такого field
                try:
                    claim = Claim(**claim_kwargs, stance=stance)
                except TypeError:
                    # Old Claim model без stance — без нього
                    claim = Claim(**claim_kwargs)
            else:
                claim = Claim(**claim_kwargs)

            claims.append(claim)
        return claims
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.debug(f"JSON parse failed: {e}")
        return []


def _extract_json_object(text: str) -> Optional[dict]:
    """
    Витягнути перший top-level {...} JSON-обʼєкт із довільного тексту
    (працює і коли LLM обгортає відповідь у ```json fences```).
    """
    if not text:
        return None
    n = len(text)
    i = 0
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_str = False
        esc = False
        for j in range(i, n):
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[i:j + 1]
                    try:
                        return json.loads(candidate)
                    except (json.JSONDecodeError, ValueError):
                        break  # try next "{"
        i += 1
    return None


def _parse_claims_fallback(response_text: str, original_text: str) -> list[Claim]:
    """
    Fallback: якщо LLM повернув не-JSON, спробувати витягти з bullets/numbered list.
    """
    if not response_text:
        return []

    lines = response_text.split("\n")
    claims = []
    for line in lines:
        line = line.strip()
        line = re.sub(r'^[\d\.\-\*•]+\s*', '', line)
        line = line.strip("\"' ")
        if len(line) < 15 or len(line) > 300:
            continue
        if not re.search(r'\b(is|was|are|were|has|have|had|causes?|makes?|says?|claims?|shows?)\b', line, re.IGNORECASE):
            continue
        try:
            claim = Claim(
                text=line,
                claim_type="statement",
                entities=[],
                verifiable=True,
                original_text=original_text,
            )
        except Exception as e:
            logger.warning(f"Cannot create Claim: {e}")
            continue
        claims.append(claim)
        if len(claims) >= 3:
            break

    return claims


# ──────────────────────────────────────────────────────────────────────────
# Synchronous helper for non-async contexts
# ──────────────────────────────────────────────────────────────────────────

def extract_claims_sync(text: str, max_claims: int = 3) -> list[Claim]:
    """Synchronous wrapper для скриптів. Use extract_claims() in async."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError("Use extract_claims() directly in async context")
    except RuntimeError:
        pass
    return asyncio.run(extract_claims(text, max_claims))
