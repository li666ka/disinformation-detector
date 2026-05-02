"""
Fact-check integration via Google Fact Check Tools API.

Docs: https://developers.google.com/fact-check/tools/api/reference/rest
"""
import os
import re
import hashlib
import logging
from difflib import SequenceMatcher
from typing import Optional
import requests

log = logging.getLogger("fact-check")

GOOGLE_FACT_CHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

FAKE_KEYWORDS = [
    "false", "fake", "hoax", "fabrica", "incorrect", "pants on fire",
    "misleading", "manipulated", "out of context", "scam", "altered",
    "inaccurate", "no evidence", "unsupported", "wrong", "myth",
    "debunked", "not true", "untrue", "spurious", "satire",
    "miscaptioned", "doctored",
]

REAL_KEYWORDS = [
    "true", "accurate", "correct", "verified", "confirmed",
    "factual", "reliable", "supported",
]

MIXED_KEYWORDS = [
    "mixed", "half", "partly", "partially", "mostly mixed",
    "needs context", "missing context", "outdated",
]

NOISE_PREFIXES = [
    "BREAKING", "ALERT", "URGENT", "JUST IN",
    "BREAKING NEWS", "DEVELOPING", "EXCLUSIVE",
    "WATCH", "VIDEO", "PHOTO", "PICS",
    "MUST READ", "MUST WATCH", "MUST SEE",
    "RT IF", "RETWEET IF",
]

_cache: dict[str, dict] = {}
_CACHE_MAX_SIZE = 1000


def _get_api_key() -> str:
    return os.getenv("GOOGLE_FACT_CHECK_API_KEY", "").strip()


def normalize_rating(rating: str) -> str:
    """
    Нормалізує fact-check rating з різних fact-checkers до FAKE/REAL/MIXED.
    """
    if not rating:
        return "MIXED"

    r = rating.lower().strip()

    if any(kw in r for kw in MIXED_KEYWORDS):
        if "true" in r and "mostly" in r:
            return "REAL"
        if "false" in r and "mostly" in r:
            return "FAKE"
        return "MIXED"

    if any(kw in r for kw in FAKE_KEYWORDS):
        return "FAKE"
    if any(kw in r for kw in REAL_KEYWORDS):
        return "REAL"

    return "MIXED"


def extract_claim_text(post_text: str, max_chars: int = 200) -> str:
    """
    Витягує найважливіший фрагмент з посту для пошуку у fact-check API.

    Steps:
      1. Видалити emoji та non-ASCII characters
      2. Видалити URLs (http/https/www)
      3. Видалити @mentions
      4. Хештеги → слова без #
      5. Прибрати noise prefixes (BREAKING:, ALERT, etc.)
      6. Нормалізувати ALL-CAPS слова (>=4 chars) → lowercase
      7. Стиснути repeated punctuation (!!! → !)
      8. Squeeze whitespace
      9. Обрізати до max_chars по word boundary
    """
    if not post_text:
        return ""

    text = post_text

    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)

    text = re.sub(r"@[\w.-]+", " ", text)

    text = re.sub(r"#(\w+)", r"\1", text)

    for prefix in NOISE_PREFIXES:
        pattern = rf"\b{re.escape(prefix)}\b[:\s]*"
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    def _normalize_caps(match: re.Match) -> str:
        word = match.group(0)
        if len(word) >= 4 and word.isupper():
            return word.lower()
        return word

    text = re.sub(r"\b[A-Z]+\b", _normalize_caps, text)

    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"\.{4,}", "...", text)

    text = re.sub(r"\s+", " ", text).strip()

    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.7:
        truncated = truncated[:last_space]
    return truncated.strip()


def _cache_key(claim_text: str, language: str) -> str:
    return hashlib.md5(f"{claim_text}|{language}".encode()).hexdigest()


def fact_check_claim(
    claim_text: str,
    language: str = "en",
    max_results: int = 5,
) -> dict:
    """Запит до Google Fact Check Tools API. Cached in-memory."""
    api_key = _get_api_key()
    if not api_key:
        return {
            "found": False,
            "error": "GOOGLE_FACT_CHECK_API_KEY not configured",
            "verdict_normalized": "UNKNOWN",
        }

    if not claim_text or len(claim_text.strip()) < 10:
        return {
            "found": False,
            "error": "Claim text too short",
            "verdict_normalized": "UNKNOWN",
        }

    ck = _cache_key(claim_text, language)
    if ck in _cache:
        log.debug(f"Cache hit: {claim_text[:50]}...")
        return _cache[ck]

    params = {
        "query": claim_text,
        "languageCode": language,
        "key": api_key,
        "pageSize": max_results,
    }

    try:
        response = requests.get(GOOGLE_FACT_CHECK_URL, params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.HTTPError as e:
        log.error(f"Google Fact Check API HTTP error: {e}")
        return {
            "found": False,
            "error": f"API error: {e.response.status_code if e.response else 'unknown'}",
            "verdict_normalized": "UNKNOWN",
        }
    except Exception as e:
        log.error(f"Google Fact Check API failed: {e}")
        return {
            "found": False,
            "error": str(e),
            "verdict_normalized": "UNKNOWN",
        }

    claims = data.get("claims", [])
    if not claims:
        result = {
            "found": False,
            "claims": [],
            "verdict_normalized": "UNKNOWN",
            "message": "No fact-checks found for this claim",
        }
        _store_cache(ck, result)
        return result

    SIM_THRESHOLD = 0.4
    best_claim = None
    best_sim = 0.0
    for cand in claims:
        cand_text = cand.get("text", "") or ""
        sim = _similarity(claim_text, cand_text)
        if sim > best_sim and cand.get("claimReview"):
            best_sim = sim
            best_claim = cand

    if best_claim is None or best_sim < SIM_THRESHOLD:
        result = {
            "found": False,
            "claims": claims,
            "verdict_normalized": "UNKNOWN",
            "message": f"Low similarity matches only (best={best_sim:.2f})",
        }
        _store_cache(ck, result)
        return result

    top_claim = best_claim
    reviews = top_claim.get("claimReview", [])
    review = reviews[0]
    rating = review.get("textualRating", "")
    publisher = review.get("publisher", {}).get("name", "Unknown")

    result = {
        "found": True,
        "claims": claims,
        "top_claim": top_claim,
        "claim_text_matched": top_claim.get("text", ""),
        "match_similarity": best_sim,
        "verdict": rating,
        "verdict_normalized": normalize_rating(rating),
        "publisher": publisher,
        "url": review.get("url", ""),
        "review_date": review.get("reviewDate", ""),
        "review_title": review.get("title", ""),
    }
    _store_cache(ck, result)
    return result


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _store_cache(key: str, value: dict) -> None:
    if len(_cache) >= _CACHE_MAX_SIZE:
        _cache.pop(next(iter(_cache)))
    _cache[key] = value


def _effective_author_verdict(stance: str, fc_verdict: str) -> str:
    """
    Effective author position = stance × fact-check verdict.

    Якщо claim is FAKE (per fact-checkers) і автор:
      - supports → автор spreading misinfo (FAKE)
      - refutes → автор корректно спростовує (REAL)
      - neutral → MIXED

    Якщо claim is REAL і автор:
      - supports → автор корректно стверджує (REAL)
      - refutes → автор спростовує true claim (FAKE)
      - neutral → MIXED
    """
    if fc_verdict == "FAKE":
        if stance == "supports":
            return "FAKE"
        if stance == "refutes":
            return "REAL"
        return "MIXED"
    if fc_verdict == "REAL":
        if stance == "supports":
            return "REAL"
        if stance == "refutes":
            return "FAKE"
        return "MIXED"
    return "MIXED"


def verify_post(
    post_text: str,
    model_label: Optional[str] = None,
    model_confidence: Optional[float] = None,
    language: str = "en",
    use_llm_extraction: bool = True,
) -> dict:
    """
    Three-stage verification:
      1. Extract claims з stance detection (LLM/regex fallback).
      2. Query Google Fact Check для canonical form кожного claim.
      3. Compute effective_author_verdict = stance × fact-check verdict.
         Aggregate by majority vote → overall verdict.
         Compare з model output.
    """
    from api.claim_extractor import extract_claims

    extraction = extract_claims(post_text, use_llm=use_llm_extraction)
    claim_items: list[dict] = extraction["claims"]
    extraction_method: str = extraction["method"]

    if not claim_items:
        return {
            "fact_check_found": False,
            "extraction_method": extraction_method,
            "claims_extracted": [],
            "claims_results": [],
            "verdict_normalized": "UNKNOWN",
            "verdict": None,
            "publisher": None,
            "url": None,
            "review_date": None,
            "claim_text_matched": None,
            "claim_query_used": "",
            "claims_total": 0,
            "claims_found": 0,
            "fake_count": 0,
            "real_count": 0,
            "mixed_count": 0,
            "model_label": model_label,
            "model_confidence": model_confidence,
            "comparison_status": "NO_DATA",
            "match": None,
            "message": "No claims extracted from post",
        }

    fc_results = [fact_check_claim(item["claim"], language=language) for item in claim_items]

    claims_with_results: list[dict] = []
    found_any = False
    author_position_counts = {"REAL": 0, "FAKE": 0, "MIXED": 0, "UNKNOWN": 0}

    for item, fc in zip(claim_items, fc_results):
        claim = item["claim"]
        stance = item["stance"]
        author_verdict_initial = item["author_verdict"]

        if fc.get("found"):
            found_any = True
            fc_verdict = fc.get("verdict_normalized", "UNKNOWN")
            effective = _effective_author_verdict(stance, fc_verdict)
        else:
            effective = "UNKNOWN"

        author_position_counts[effective] += 1

        claims_with_results.append({
            "claim": claim,
            "stance": stance,
            "author_verdict_initial": author_verdict_initial,
            "found": fc.get("found", False),
            "verdict": fc.get("verdict"),
            "verdict_normalized": fc.get("verdict_normalized", "UNKNOWN"),
            "effective_author_verdict": effective,
            "publisher": fc.get("publisher"),
            "url": fc.get("url"),
            "review_date": fc.get("review_date"),
            "review_title": fc.get("review_title"),
            "claim_text_matched": fc.get("claim_text_matched"),
            "match_similarity": fc.get("match_similarity"),
        })

    if not found_any:
        overall_verdict = "UNKNOWN"
    elif author_position_counts["FAKE"] > author_position_counts["REAL"]:
        overall_verdict = "FAKE"
    elif author_position_counts["REAL"] > author_position_counts["FAKE"]:
        overall_verdict = "REAL"
    else:
        overall_verdict = "MIXED"

    top_result = next(
        (r for r in claims_with_results if r["found"]),
        claims_with_results[0] if claims_with_results else {},
    )

    comparison: dict = {
        "fact_check_found": found_any,
        "extraction_method": extraction_method,
        "claims_extracted": [item["claim"] for item in claim_items],
        "claims_results": claims_with_results,
        "verdict_normalized": overall_verdict,
        "verdict": top_result.get("verdict"),
        "publisher": top_result.get("publisher"),
        "url": top_result.get("url"),
        "review_date": top_result.get("review_date"),
        "review_title": top_result.get("review_title"),
        "claim_text_matched": top_result.get("claim_text_matched"),
        "claim_query_used": top_result.get("claim", claim_items[0]["claim"] if claim_items else ""),
        "claims_total": len(claim_items),
        "claims_found": sum(1 for r in claims_with_results if r["found"]),
        "fake_count": author_position_counts["FAKE"],
        "real_count": author_position_counts["REAL"],
        "mixed_count": author_position_counts["MIXED"],
        "model_label": model_label,
        "model_confidence": model_confidence,
    }

    if not found_any:
        comparison["comparison_status"] = "NO_DATA"
        comparison["match"] = None
    elif overall_verdict in ("MIXED", "UNKNOWN"):
        comparison["comparison_status"] = "MIXED"
        comparison["match"] = None
    elif model_label is None:
        comparison["comparison_status"] = "NO_MODEL"
        comparison["match"] = None
    elif overall_verdict == model_label:
        comparison["comparison_status"] = "MATCH"
        comparison["match"] = True
    else:
        comparison["comparison_status"] = "MISMATCH"
        comparison["match"] = False

    return comparison
