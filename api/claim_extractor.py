"""
Claim extraction for fact_check.py — синхронний інтерфейс.

Витягує перевірювані твердження + stance автора через Claude CLI subprocess.
Працює напряму через `_call_claude_cli` з api.llm_predictor (без Gemini shim).

Контракт (для fact_check.verify_post):
    extract_claims(text: str, use_llm: bool = True) -> dict
    повертає {
        "claims": [
            {"claim": str, "stance": "supports"|"refutes"|"neutral",
             "author_verdict": "REAL"|"FAKE"|"MIXED"},
            ...
        ],
        "method": "llm" | "fallback"
    }
"""
from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)


CLAIM_EXTRACTION_PROMPT = """You are an expert fact-checker. Extract verifiable factual claims from the given text along with the author's stance toward each claim.

A "claim" is a statement that asserts a fact about the world and can be verified or refuted with evidence. It is NOT an opinion, emotion, question, personal experience, or pure speculation.

For each claim, return:
- claim: clear standalone sentence (rewrite for clarity, resolve pronouns; ≤180 chars)
- stance: one of
    "supports"  — author asserts the claim is TRUE
    "refutes"   — author asserts the claim is FALSE (e.g., "X does NOT cause Y")
    "neutral"   — author just mentions / quotes without taking a position

If the text contains NO verifiable claims, return an empty list.
Return AT MOST 3 claims (most central ones).

Respond ONLY with valid JSON, no prose:
{"claims": [{"claim": "...", "stance": "supports"}, ...]}
"""


def extract_claims(text: str, use_llm: bool = True) -> dict:
    """
    Sync extraction. Try Claude CLI first; fall back to regex on any failure.
    """
    if not text or len(text.strip()) < 10:
        return {"claims": [], "method": "fallback"}

    text = text.strip()[:2000]

    if use_llm:
        llm_claims = _extract_via_claude(text)
        if llm_claims:
            return {"claims": llm_claims, "method": "llm"}

    return {"claims": _extract_via_regex(text), "method": "fallback"}


def _extract_via_claude(text: str) -> list[dict]:
    try:
        from api.llm_predictor import _call_claude_cli, ClaudeCLIError, DEFAULT_BASE_MODEL
    except ImportError as e:
        logger.error(f"llm_predictor unavailable: {e}")
        return []

    full_prompt = f"{CLAIM_EXTRACTION_PROMPT}\n\nText to analyze:\n\n{text}"

    try:
        raw = _call_claude_cli(full_prompt, model=DEFAULT_BASE_MODEL)
    except ClaudeCLIError as e:
        logger.warning(f"Claude CLI failed for claim extraction: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error calling Claude CLI: {e}")
        return []

    return _parse_llm_response(raw)


def _parse_llm_response(raw_text: str) -> list[dict]:
    if not raw_text:
        return []

    data = _extract_json_object(raw_text)
    if data is None:
        return []

    raw_claims = data.get("claims", [])
    if not isinstance(raw_claims, list):
        return []

    out: list[dict] = []
    for rc in raw_claims:
        if not isinstance(rc, dict):
            continue
        claim_text = (rc.get("claim") or rc.get("text") or "").strip()
        if not claim_text or len(claim_text) < 10:
            continue
        stance = str(rc.get("stance", "neutral")).lower().strip()
        if stance not in ("supports", "refutes", "neutral"):
            stance = "neutral"
        out.append({
            "claim": claim_text[:300],
            "stance": stance,
            "author_verdict": _stance_to_author_verdict(stance),
        })
        if len(out) >= 3:
            break
    return out


def _extract_json_object(text: str) -> dict | None:
    """Витягти перший balanced top-level {...} JSON-обʼєкт (терпить ```json fences)."""
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
                        break
        i += 1
    return None


def _stance_to_author_verdict(stance: str) -> str:
    """
    Initial author verdict (BEFORE fact-check):
      supports → REAL  (author treats claim as true)
      refutes  → FAKE  (author treats claim as false)
      neutral  → MIXED (author neither asserts nor denies)
    fact_check.py later corrects this against external fact-checker rating.
    """
    if stance == "supports":
        return "REAL"
    if stance == "refutes":
        return "FAKE"
    return "MIXED"


_FACTUAL_VERB_RE = re.compile(
    r'\b(is|are|was|were|has|have|had|will|did|does|do|cause|causes|caused|'
    r'kills?|killed|spreads?|contains?|proves?|shows?|reveals?|claims?|reports?|'
    r'announces?|confirms?|denies?|denied)\b',
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r'\b(not|no|never|isn\'t|aren\'t|wasn\'t|weren\'t|doesn\'t|don\'t|didn\'t|'
    r'cannot|can\'t|won\'t|denies|denied|debunked|false)\b',
    re.IGNORECASE,
)


def _extract_via_regex(text: str) -> list[dict]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    claims: list[dict] = []
    for s in sentences:
        s = s.strip().strip('"\'')
        if len(s) < 20 or len(s) > 280:
            continue
        if not _FACTUAL_VERB_RE.search(s):
            continue
        stance = "refutes" if _NEGATION_RE.search(s) else "supports"
        claims.append({
            "claim": s,
            "stance": stance,
            "author_verdict": _stance_to_author_verdict(stance),
        })
        if len(claims) >= 3:
            break
    return claims
