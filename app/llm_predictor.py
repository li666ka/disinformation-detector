# app/llm_predictor.py
"""
LLM-based zero-shot classification via Gemini API.

Two modes:
  - single:  1 request, temperature=0, deterministic
  - bagging: N=3 requests, temperature=0.7, majority voting
"""

import os
import re
import json
import logging
from collections import Counter

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

SYSTEM_PROMPT = (
    "You are an expert media literacy analyst. Your task is to classify text as either "
    "REAL (factual, credible information) or FAKE (misinformation, disinformation, propaganda).\n\n"
    "Analyze the text for: factual claims without evidence, emotional manipulation, "
    "anonymous authority references, conspiracy framing, clickbait patterns, logical fallacies.\n\n"
    "Respond ONLY with valid JSON:\n"
    '{"label": "REAL" or "FAKE", "confidence": 0.0-1.0, "reason": "brief explanation"}'
)


def _build_user_prompt(text: str, feature_values: dict | None = None) -> str:
    prompt = f"Classify this text:\n\n{text[:2000]}"
    if feature_values:
        non_zero = {k: round(v, 4) for k, v in feature_values.items() if v != 0}
        if non_zero:
            prompt += f"\n\nComputed linguistic features:\n{json.dumps(non_zero, indent=2)}"
    return prompt


def _parse_response(text: str) -> dict:
    """Extract JSON from Gemini response."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            label = data.get("label", "UNCERTAIN").upper()
            if label not in ("REAL", "FAKE"):
                label = "UNCERTAIN"
            return {
                "label": label,
                "confidence": float(data.get("confidence", 0.5)),
                "reason": data.get("reason", ""),
            }
        except (json.JSONDecodeError, ValueError):
            pass
    return {"label": "UNCERTAIN", "confidence": 0.5, "reason": "parse_error"}


def _get_client():
    """Create Gemini client."""
    from google import genai

    api_key = GEMINI_API_KEY
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set. Configure it in .env file.")
    return genai.Client(api_key=api_key)


def predict_single(
    text: str,
    feature_values: dict | None = None,
) -> dict:
    """Single deterministic classification (temperature=0)."""
    from google.genai import types

    client = _get_client()
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=_build_user_prompt(text, feature_values),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,
            max_output_tokens=300,
        ),
    )
    response_text = response.text.strip()
    result = _parse_response(response_text)
    logger.info(f"LLM single: label={result['label']}, conf={result['confidence']:.2f}")
    return result


def predict_bagging(
    text: str,
    feature_values: dict | None = None,
    n_calls: int = 3,
) -> dict:
    """
    Bagging: N calls with temperature=0.7, majority voting.
    Confidence = fraction of votes for winning label * avg confidence of those votes.
    """
    from google.genai import types

    client = _get_client()
    user_prompt = _build_user_prompt(text, feature_values)

    results = []
    for i in range(n_calls):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.7,
                    max_output_tokens=300,
                ),
            )
            parsed = _parse_response(response.text.strip())
            results.append(parsed)
        except Exception as e:
            logger.warning(f"LLM bagging call {i+1}/{n_calls} failed: {e}")

    if not results:
        raise RuntimeError("All LLM bagging calls failed")

    # Majority voting
    labels = [r["label"] for r in results if r["label"] in ("REAL", "FAKE")]
    if not labels:
        return {"label": "UNCERTAIN", "confidence": 0.5, "reason": "all_uncertain"}

    counter = Counter(labels)
    winner, winner_count = counter.most_common(1)[0]
    vote_fraction = winner_count / len(labels)

    # Average confidence of winning label votes
    winner_confs = [r["confidence"] for r in results if r["label"] == winner]
    avg_conf = sum(winner_confs) / len(winner_confs)

    confidence = vote_fraction * avg_conf
    reasons = [r["reason"] for r in results if r["label"] == winner and r["reason"]]

    logger.info(f"LLM bagging: label={winner}, votes={counter}, conf={confidence:.2f}")
    return {
        "label": winner,
        "confidence": round(confidence, 4),
        "reason": reasons[0] if reasons else "",
        "votes": dict(counter),
        "n_calls": len(results),
    }


def predict(
    text: str,
    mode: str = "single",
    feature_values: dict | None = None,
) -> dict:
    """
    Dispatch to single or bagging mode.
    Returns: {label, confidence, reason, ...}
    """
    if mode == "bagging":
        return predict_bagging(text, feature_values)
    return predict_single(text, feature_values)
