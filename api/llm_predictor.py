# api/llm_predictor.py
"""
LLM classification via Gemini API — preset-driven.

predict_with_preset(text, preset_config) — new preset-based entry point.
predict(...) — legacy wrapper, kept for backward compat during migration.

Models (verified April 2026, free tier):
  gemini-2.5-flash-lite  (15 RPM / 1000 RPD)
  gemini-2.5-flash       (10 RPM /  250 RPD)
  gemini-2.5-pro         ( 5 RPM /  100 RPD)
"""
import os
import re
import json
import time
import logging
from collections import Counter

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

DEFAULT_BASE_MODEL = "gemini-2.5-flash-lite"

# Fallback chain: if requested model hits quota, try progressively smaller.
FALLBACK_CHAIN = {
    "gemini-2.5-pro":        ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
    "gemini-2.5-flash":      ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
    "gemini-2.5-flash-lite": ["gemini-2.5-flash-lite"],
}


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert media literacy analyst. Your task is to classify text as either "
    "REAL (factual, credible information) or FAKE (misinformation, disinformation, propaganda).\n\n"
    "Analyze the text for: factual claims without evidence, emotional manipulation, "
    "anonymous authority references, conspiracy framing, clickbait patterns, logical fallacies.\n\n"
    "Respond ONLY with valid JSON:\n"
    '{"label": "REAL" or "FAKE", "confidence": 0.0-1.0, "reason": "brief explanation"}'
)

DEFAULT_COT_INSTRUCTION = (
    "Before giving your final answer, reason step by step:\n"
    "1. What are the key claims in this text?\n"
    "2. Are these claims verifiable from reliable sources?\n"
    "3. Is there emotional manipulation or sensationalism?\n"
    "4. Does the language match credible journalism or clickbait?\n"
    "After this reasoning, output ONLY the final JSON with label, confidence and reason."
)


def _get_client():
    """Lazy-init Gemini client."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise RuntimeError(
            "google-genai SDK not installed. Run: pip install google-genai"
        ) from e

    api_key = GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set. Configure it in .env file.")
    return genai.Client(api_key=api_key), types


def _parse_response(text: str) -> dict:
    """Parse Gemini's JSON response into {label, confidence, reason}."""
    if not text:
        return {"label": "UNCERTAIN", "confidence": 0.5, "reason": "empty_response"}

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

    upper = text.upper()
    if "FAKE" in upper and "REAL" not in upper:
        return {"label": "FAKE", "confidence": 0.6, "reason": "fallback_regex"}
    if "REAL" in upper and "FAKE" not in upper:
        return {"label": "REAL", "confidence": 0.6, "reason": "fallback_regex"}
    return {"label": "UNCERTAIN", "confidence": 0.5, "reason": "parse_error"}


def _build_user_prompt_zero_shot(text: str) -> str:
    return f"Classify this text:\n\n{text[:2000]}"


def _build_user_prompt_few_shot(text: str, examples: list[dict]) -> str:
    parts = ["Here are some labeled examples to guide your classification:\n"]
    for i, ex in enumerate(examples, 1):
        parts.append(f"Example {i} ({ex['label']}): {ex['text'][:500]}\n")
    parts.append(f"\nNow classify this new text:\n\n{text[:2000]}")
    return "\n".join(parts)


def _build_user_prompt_cot(text: str, cot_instruction: str) -> str:
    return f"{cot_instruction}\n\nText to classify:\n\n{text[:2000]}"


def _call_gemini_raw(
    client,
    types,
    *,
    base_model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_output_tokens: int,
) -> tuple[str, str]:
    """One Gemini call with fallback. Returns (raw_text, model_used).

    Unlike _call_gemini, does NOT parse the response — returns the raw text
    so callers (claim extraction, stance detection, etc.) can parse it themselves.
    """
    chain = FALLBACK_CHAIN.get(base_model, [DEFAULT_BASE_MODEL])

    for model_id in chain:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                    ),
                )
                if not response or not response.text:
                    logger.warning(f"{model_id}: empty response")
                    break
                return response.text.strip(), model_id
            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
                    delay_match = re.search(r'retryDelay.*?(\d+)', err)
                    wait = int(delay_match.group(1)) + 2 if delay_match else 15
                    if attempt < 2:
                        logger.warning(f"{model_id}: rate limited, retry in {wait}s")
                        time.sleep(wait)
                    else:
                        logger.warning(f"{model_id}: quota exhausted, trying next model")
                        break
                elif "404" in err or "not found" in err.lower():
                    logger.warning(f"{model_id}: not available, trying next")
                    break
                else:
                    logger.warning(f"{model_id}: unexpected error: {e}")
                    break

    return "", "none"


def _call_gemini(
    client,
    types,
    *,
    base_model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_output_tokens: int,
) -> tuple[dict, str]:
    """One Gemini call with fallback. Returns (parsed_result, model_used)."""
    raw_text, model_used = _call_gemini_raw(
        client, types,
        base_model=base_model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    if not raw_text:
        return (
            {"label": "UNCERTAIN", "confidence": 0.5, "reason": "all_models_exhausted"},
            model_used,
        )
    return _parse_response(raw_text), model_used


# Backward-compat alias for /evaluate endpoint which calls _call_with_fallback
def _call_with_fallback(client, types, contents, temperature=0, max_tokens=300):
    """Legacy wrapper for main.py::evaluate_llm."""
    return _call_gemini(
        client, types,
        base_model=DEFAULT_BASE_MODEL,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        user_prompt=contents,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )


def predict_with_preset(text: str, preset_config: dict) -> dict:
    """
    Classify text using a stored LLM preset config.

    preset_config keys:
        base_model, mode (zero_shot|few_shot|cot|bagging),
        system_prompt, temperature, max_output_tokens,
        few_shot_examples (for few_shot), cot_instruction (for cot),
        bagging_n_calls (for bagging)

    Returns {label, confidence, reason, base_model_used, mode, n_calls, votes?}.
    """
    client, types = _get_client()

    mode = preset_config.get("mode", "zero_shot")
    base_model = preset_config.get("base_model", DEFAULT_BASE_MODEL)
    system_prompt = preset_config.get("system_prompt") or DEFAULT_SYSTEM_PROMPT
    temperature = float(preset_config.get("temperature", 0.0))
    max_output_tokens = int(preset_config.get("max_output_tokens", 200))

    if mode == "zero_shot":
        user_prompt = _build_user_prompt_zero_shot(text)
        return _run_single(client, types, base_model, system_prompt, user_prompt,
                           temperature, max_output_tokens, mode)

    elif mode == "few_shot":
        examples = preset_config.get("few_shot_examples") or []
        if not examples:
            logger.warning("few_shot: no examples, falling back to zero_shot")
            user_prompt = _build_user_prompt_zero_shot(text)
        else:
            user_prompt = _build_user_prompt_few_shot(text, examples)
        return _run_single(client, types, base_model, system_prompt, user_prompt,
                           temperature, max_output_tokens, mode)

    elif mode == "cot":
        cot_instruction = preset_config.get("cot_instruction") or DEFAULT_COT_INSTRUCTION
        user_prompt = _build_user_prompt_cot(text, cot_instruction)
        cot_tokens = max(max_output_tokens, 500)
        return _run_single(client, types, base_model, system_prompt, user_prompt,
                           temperature, cot_tokens, mode)

    elif mode == "bagging":
        n_calls = int(preset_config.get("bagging_n_calls", 3))
        bagging_temp = max(temperature, 0.7)
        user_prompt = _build_user_prompt_zero_shot(text)
        return _run_bagging(client, types, base_model, system_prompt, user_prompt,
                            bagging_temp, max_output_tokens, n_calls)

    else:
        raise ValueError(f"Unknown mode: {mode}")


def _run_single(client, types, base_model, system_prompt, user_prompt,
                temperature, max_output_tokens, mode) -> dict:
    result, model_used = _call_gemini(
        client, types,
        base_model=base_model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    logger.info(f"LLM {mode} ({model_used}): label={result['label']}, conf={result['confidence']:.2f}")
    return {
        "label": result["label"],
        "confidence": result["confidence"],
        "reason": result.get("reason", ""),
        "base_model_used": model_used,
        "mode": mode,
        "n_calls": 1,
    }


def _run_bagging(client, types, base_model, system_prompt, user_prompt,
                 temperature, max_output_tokens, n_calls) -> dict:
    results = []
    models_used = []

    for i in range(n_calls):
        try:
            parsed, model_used = _call_gemini(
                client, types,
                base_model=base_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            results.append(parsed)
            models_used.append(model_used)
        except Exception as e:
            logger.warning(f"bagging call {i+1}/{n_calls} failed: {e}")

    if not results:
        return {
            "label": "UNCERTAIN", "confidence": 0.5, "reason": "all_bagging_failed",
            "base_model_used": "none", "mode": "bagging", "n_calls": 0,
        }

    labels = [r["label"] for r in results if r["label"] in ("REAL", "FAKE")]
    if not labels:
        return {
            "label": "UNCERTAIN", "confidence": 0.5, "reason": "all_bagging_uncertain",
            "base_model_used": models_used[0] if models_used else "none",
            "mode": "bagging", "n_calls": len(results),
        }

    counter = Counter(labels)
    winner, winner_count = counter.most_common(1)[0]
    vote_fraction = winner_count / len(labels)
    winner_confs = [r["confidence"] for r in results if r["label"] == winner]
    avg_conf = sum(winner_confs) / len(winner_confs)
    confidence = vote_fraction * avg_conf
    reasons = [r["reason"] for r in results if r["label"] == winner and r["reason"]]

    logger.info(f"LLM bagging: label={winner}, votes={dict(counter)}, conf={confidence:.2f}")
    return {
        "label": winner,
        "confidence": round(confidence, 4),
        "reason": reasons[0] if reasons else "",
        "base_model_used": models_used[0] if models_used else "none",
        "mode": "bagging",
        "n_calls": len(results),
        "votes": dict(counter),
    }


# ── Backward compat wrappers (for /evaluate endpoint) ───────────────────

def predict(text: str, mode: str = "single", feature_values: dict | None = None) -> dict:
    """Legacy wrapper. Used by main.py evaluate_llm and analyze_text LLM branch (fallback)."""
    preset = {
        "base_model": DEFAULT_BASE_MODEL,
        "mode": "bagging" if mode == "bagging" else "zero_shot",
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "temperature": 0.7 if mode == "bagging" else 0.0,
        "max_output_tokens": 200,
        "bagging_n_calls": 3,
    }
    return predict_with_preset(text, preset)


def predict_single(text: str, feature_values: dict | None = None) -> dict:
    return predict(text, mode="single")


def predict_bagging(text: str, feature_values: dict | None = None, n_calls: int = 3) -> dict:
    return predict(text, mode="bagging")
