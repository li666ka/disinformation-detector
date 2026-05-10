# api/llm_predictor.py
"""
LLM classification via Claude Code CLI — preset-driven.

Виклики Claude через subprocess `claude -p` — без API token, через Max plan OAuth.
ANTHROPIC_API_KEY не повинен бути встановлений у env (інакше біллинг піде на API).

Models (April 2026):
  claude-haiku-4-5    — найшвидша, дешева (default)
  claude-sonnet-4-6   — баланс
  claude-opus-4-7     — найрозумніша (повільніша)

Public API:
  predict_with_preset(text, preset_config) — single classification
  predict_batch_with_preset(texts, preset_config) — BATCH classification (10-15 за раз)
  predict(...) — legacy wrapper

Low-level (used by claim_extractor / verification pipeline):
  _call_claude_cli(prompt, model=...) → str  — raw Claude CLI subprocess call.
"""
import os
import re
import json
import time
import logging
import subprocess
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────

DEFAULT_BASE_MODEL = "claude-haiku-4-5"

AVAILABLE_MODELS = [
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
]

# CLI executable
CLAUDE_CLI = os.environ.get("CLAUDE_CLI_PATH", "claude")

# Timeouts (seconds)
SINGLE_TIMEOUT = 60
BATCH_TIMEOUT = 180

# Default rate limit pause (seconds between calls)
DEFAULT_PAUSE = 0.5


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert media literacy analyst. Your task is to classify text as either "
    "REAL (factual, credible information) or FAKE (misinformation, disinformation, propaganda).\n\n"
    "Analyze the text for: factual claims without evidence, emotional manipulation, "
    "anonymous authority references, conspiracy framing, clickbait patterns, logical fallacies.\n\n"
    "Respond ONLY with valid JSON:\n"
    '{"label": "REAL" or "FAKE", "confidence": 0.0-1.0, "reason": "brief explanation"}'
)

DEFAULT_BATCH_SYSTEM_PROMPT = (
    "You are an expert media literacy analyst. You will receive MULTIPLE texts in one request. "
    "For EACH text, classify as REAL or FAKE.\n\n"
    "Analyze each for: factual claims without evidence, emotional manipulation, "
    "anonymous authority references, conspiracy framing, clickbait patterns, logical fallacies.\n\n"
    "Respond ONLY with a valid JSON array. Each input has a 'Text ID N:' marker — you MUST include "
    'the matching "text_id" field in each output object so we can match input to output:\n'
    '[{"text_id": 0, "label": "FAKE", "confidence": 0.85, "reason": "brief"}, ...]'
)

DEFAULT_COT_INSTRUCTION = (
    "Before giving your final answer, reason step by step:\n"
    "1. What are the key claims in this text?\n"
    "2. Are these claims verifiable from reliable sources?\n"
    "3. Is there emotional manipulation or sensationalism?\n"
    "4. Does the language match credible journalism or clickbait?\n"
    "After this reasoning, output ONLY the final JSON with label, confidence and reason."
)


# Module-level counter for monitoring parse health.
# Read via get_parse_failure_stats() from health endpoints / dashboards.
_parse_failures = Counter()


def get_parse_failure_stats() -> dict:
    """Return cumulative parse-failure reasons for monitoring."""
    return dict(_parse_failures)


def reset_parse_failure_stats() -> None:
    _parse_failures.clear()


# ─────────────────────────────────────────────────────────────────
# CORE: Claude Code CLI invocation
# ─────────────────────────────────────────────────────────────────

class ClaudeCLIError(RuntimeError):
    """Помилка виклику claude CLI (timeout, exit_code, parse fail)."""


def _check_api_key_unset() -> None:
    """ANTHROPIC_API_KEY must NOT be set — Max-plan OAuth would be bypassed and billing flips to API."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "❌ ANTHROPIC_API_KEY встановлений у env. "
            "Видаліть його (`unset ANTHROPIC_API_KEY`) перед запуском backend, "
            "інакше calls будуть біллитись на API account замість Max plan."
        )


def _check_claude_available() -> tuple[bool, str]:
    """
    Lightweight CLI availability probe used for graceful fallback.
    Returns (is_available, error_message). Never raises.
    """
    try:
        result = subprocess.run(
            [CLAUDE_CLI, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        return False, f"'{CLAUDE_CLI}' not found in PATH"
    except subprocess.TimeoutExpired:
        return False, "claude --version timed out"
    except Exception as e:
        return False, f"unexpected error: {e}"

    if result.returncode != 0:
        return False, f"claude --version exit_code={result.returncode}"
    return True, ""


def _check_environment() -> None:
    """Hard check used by legacy callers: API key + CLI must both be OK."""
    _check_api_key_unset()
    available, error = _check_claude_available()
    if not available:
        if "not found in PATH" in error:
            raise RuntimeError(
                f"❌ `{CLAUDE_CLI}` CLI не знайдено у PATH. "
                "Встановіть Claude Code: https://claude.com/claude-code"
            )
        raise ClaudeCLIError(error)


def _call_claude_cli(
    prompt: str,
    *,
    model: str = DEFAULT_BASE_MODEL,
    timeout: int = SINGLE_TIMEOUT,
) -> str:
    """
    Один виклик `claude -p "prompt" --model X --output-format json`.
    Повертає raw text відповіді (поле `result` з outer JSON).

    Raises:
        ClaudeCLIError при будь-якій failure.
    """
    cmd = [
        CLAUDE_CLI,
        "-p", prompt,
        "--model", model,
        "--output-format", "json",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise ClaudeCLIError(f"claude -p timed out after {timeout}s")
    except FileNotFoundError:
        raise ClaudeCLIError(f"`{CLAUDE_CLI}` not in PATH")

    if result.returncode != 0:
        raise ClaudeCLIError(
            f"claude -p exit_code={result.returncode}, "
            f"stderr={result.stderr[:300]}"
        )

    if not result.stdout:
        raise ClaudeCLIError("claude -p returned empty stdout")

    # Outer JSON: {"result": "...", "session_id": "...", ...}
    try:
        outer = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise ClaudeCLIError(f"outer JSON parse failed: {e}, raw={result.stdout[:200]}")

    response_text = outer.get("result", "")
    if not response_text:
        raise ClaudeCLIError(f"no 'result' field in claude output: {result.stdout[:200]}")

    return response_text


# ─────────────────────────────────────────────────────────────────
# RESPONSE PARSING
# ─────────────────────────────────────────────────────────────────

def _fallback_result(reason: str) -> dict:
    """Standard UNCERTAIN result. Logs to _parse_failures for monitoring."""
    _parse_failures[reason] += 1
    return {"label": "UNCERTAIN", "confidence": 0.5, "reason": reason}


def _normalize_label(raw: str) -> str:
    """Coerce arbitrary text into REAL/FAKE/UNCERTAIN."""
    label = str(raw or "").upper().strip()
    if label in ("REAL", "FAKE", "UNCERTAIN"):
        return label
    if any(kw in label for kw in ("FAKE", "FALSE", "MISINFO")):
        return "FAKE"
    if any(kw in label for kw in ("REAL", "TRUE", "FACTUAL", "CREDIBLE")):
        return "REAL"
    return "UNCERTAIN"


def _parse_response(text: str) -> dict:
    """
    Parse Claude's JSON response into {label, confidence, reason}.

    Strategy:
      1. Try to locate a JSON object containing "label" → parse strictly.
      2. Try a broader {...} match.
      3. If JSON parse fails, regex-extract REAL/FAKE from plain text.
      4. Last resort: UNCERTAIN with reason set for monitoring.
    """
    if not text:
        return _fallback_result("empty_response")

    match = re.search(r'\{[^{}]*"label"[^{}]*\}', text, re.DOTALL)
    if not match:
        match = re.search(r'\{.+\}', text, re.DOTALL)

    if match:
        try:
            data = json.loads(match.group(0))
            label = _normalize_label(data.get("label", "UNCERTAIN"))

            try:
                confidence = float(data.get("confidence", 0.5))
            except (ValueError, TypeError):
                confidence = 0.5
            confidence = max(0.0, min(1.0, confidence))

            reason = str(data.get("reason", "")).strip()[:300]
            return {"label": label, "confidence": confidence, "reason": reason}
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("JSON parse failed: %s\nRaw output:\n%s", e, text[:500])

    # Plain-text fallback — no JSON or JSON broken; extract bare label
    label_match = re.search(r'\b(FAKE|REAL)\b', text, re.IGNORECASE)
    if label_match:
        extracted = label_match.group(1).upper()
        logger.warning("Extracted label '%s' from non-JSON response", extracted)
        _parse_failures["extracted_from_non_json_response"] += 1
        return {
            "label": extracted,
            "confidence": 0.5,
            "reason": "extracted_from_non_json_response",
        }

    logger.error("Could not parse label from response:\n%s", text[:500])
    return _fallback_result("parse_failed")


def _parse_batch_response(text: str, expected_count: int) -> list[Optional[dict]]:
    """
    Parse batch response — JSON array of {text_id, label, confidence, reason}.

    Returns list of length `expected_count`. Slots without a matching text_id are None;
    callers (predict_batch_with_preset) fill these with UNCERTAIN.

    Backward compat: if items lack `text_id` but have legacy `id` (1-based) or no id at all,
    we fall back to that ordering rather than dropping responses.
    """
    if not text:
        logger.warning("Empty batch response")
        _parse_failures["batch_empty_response"] += 1
        return [None] * expected_count

    match = re.search(r'\[.+\]', text, re.DOTALL)
    if not match:
        logger.warning("No JSON array in batch response: %s", text[:200])
        _parse_failures["batch_no_array"] += 1
        return [None] * expected_count

    try:
        arr = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        logger.warning("Batch JSON parse failed: %s, raw=%s", e, match.group(0)[:300])
        _parse_failures["batch_json_invalid"] += 1
        return [None] * expected_count

    if not isinstance(arr, list):
        logger.warning("Batch response is not a list: %s", type(arr).__name__)
        _parse_failures["batch_not_list"] += 1
        return [None] * expected_count

    # Detect ordering source: prefer text_id (0-based), fall back to id (1-based), else positional.
    has_text_id = any(isinstance(x, dict) and "text_id" in x for x in arr)
    has_legacy_id = any(isinstance(x, dict) and "id" in x for x in arr)

    results_by_id: dict[int, dict] = {}
    positional: list[Optional[dict]] = []

    for pos, item in enumerate(arr):
        if not isinstance(item, dict):
            logger.warning("Non-dict batch item at pos=%d: %r", pos, item)
            _parse_failures["batch_non_dict_item"] += 1
            positional.append(None)
            continue

        parsed = {
            "label": _normalize_label(item.get("label", "UNCERTAIN")),
            "confidence": _coerce_confidence(item.get("confidence", 0.5)),
            "reason": str(item.get("reason", ""))[:200],
        }

        if has_text_id:
            tid = item.get("text_id")
            if tid is None:
                logger.warning("Batch item missing text_id: %r", item)
                _parse_failures["batch_missing_text_id"] += 1
                positional.append(parsed)
                continue
            try:
                results_by_id[int(tid)] = parsed
            except (ValueError, TypeError):
                logger.warning("Batch item invalid text_id=%r", tid)
                _parse_failures["batch_invalid_text_id"] += 1
        elif has_legacy_id:
            lid = item.get("id")
            try:
                # Legacy id is 1-based — convert to 0-based
                results_by_id[int(lid) - 1] = parsed
            except (ValueError, TypeError):
                positional.append(parsed)
        else:
            positional.append(parsed)

    # Reassemble in order
    if results_by_id:
        ordered: list[Optional[dict]] = []
        for i in range(expected_count):
            if i in results_by_id:
                ordered.append(results_by_id[i])
            else:
                logger.warning("Missing batch result for text_id=%d", i)
                _parse_failures["batch_missing_index"] += 1
                ordered.append(None)
        return ordered

    # Pure positional fallback
    ordered = positional[:expected_count]
    while len(ordered) < expected_count:
        _parse_failures["batch_short_response"] += 1
        ordered.append(None)
    return ordered


def _coerce_confidence(value) -> float:
    try:
        c = float(value)
    except (ValueError, TypeError):
        return 0.5
    return max(0.0, min(1.0, c))


# ─────────────────────────────────────────────────────────────────
# PROMPT BUILDERS
# ─────────────────────────────────────────────────────────────────

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


def _build_batch_user_prompt(texts: list[str], examples: Optional[list[dict]] = None) -> str:
    """
    Build prompt for batch classification.

    Each text gets a `Text ID N:` marker (0-based) and the LLM must echo `text_id` back
    so we can validate ordering in _parse_batch_response.
    """
    parts = []

    if examples:
        parts.append("Reference examples:\n")
        for ex in examples[:5]:
            parts.append(f"  {ex['label']}: {ex['text'][:300]}")
        parts.append("")

    parts.append(
        f"Classify each of the following {len(texts)} texts. "
        "IMPORTANT: include a 'text_id' field in each response object that matches "
        "the 'Text ID' marker — this is how we pair output to input.\n"
    )
    parts.append("Texts:\n")

    for i, t in enumerate(texts):
        # If contains SOCIAL CONTEXT — preserve structure (newlines matter)
        if "[SOCIAL CONTEXT]" in t or "[ARTICLE]" in t:
            cleaned = t[:4000].strip()
        else:
            cleaned = t[:1500].replace("\n", " ").strip()
        parts.append(f'Text ID {i}: """{cleaned}"""\n')

    parts.append(
        f"\nRespond with a JSON array of {len(texts)} objects, each shaped: "
        '{"text_id": N, "label": "FAKE"|"REAL", "confidence": 0.0-1.0, "reason": "brief"}'
    )
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────
# PUBLIC API: Single prediction
# ─────────────────────────────────────────────────────────────────

def predict_with_preset(text: str, preset_config: dict) -> dict:
    """
    Classify text using a stored LLM preset config.

    preset_config keys:
        base_model: claude-haiku-4-5 | claude-sonnet-4-6 | claude-opus-4-7
        mode: zero_shot | few_shot | cot | bagging
        system_prompt: (optional) custom system prompt
        temperature: 0.0-1.0 (НЕ використовується у CLI, але зберігається у preset)
        max_output_tokens: (НЕ використовується у CLI)
        few_shot_examples: list of {text, label} (для few_shot)
        cot_instruction: custom CoT prompt (для cot)
        bagging_n_calls: int (для bagging)

    Returns: {label, confidence, reason, base_model_used, mode, n_calls, votes?}
    """
    _check_api_key_unset()

    mode = preset_config.get("mode", "zero_shot")
    base_model = preset_config.get("base_model", DEFAULT_BASE_MODEL)

    available, error = _check_claude_available()
    if not available:
        logger.error("Claude CLI unavailable: %s", error)
        return {
            "label": "UNCERTAIN",
            "confidence": 0.5,
            "reason": f"claude_cli_unavailable: {error}",
            "base_model_used": "none",
            "mode": mode,
            "n_calls": 0,
        }

    system_prompt = preset_config.get("system_prompt") or DEFAULT_SYSTEM_PROMPT

    # Build user prompt based on mode
    if mode == "zero_shot":
        user_prompt = _build_user_prompt_zero_shot(text)
    elif mode == "few_shot":
        examples = preset_config.get("few_shot_examples") or []
        if not examples:
            logger.warning("few_shot: no examples, falling back to zero_shot")
            user_prompt = _build_user_prompt_zero_shot(text)
        else:
            user_prompt = _build_user_prompt_few_shot(text, examples)
    elif mode == "cot":
        cot_instruction = preset_config.get("cot_instruction") or DEFAULT_COT_INSTRUCTION
        user_prompt = _build_user_prompt_cot(text, cot_instruction)
    elif mode == "bagging":
        n_calls = int(preset_config.get("bagging_n_calls", 3))
        user_prompt = _build_user_prompt_zero_shot(text)
        return _run_bagging(base_model, system_prompt, user_prompt, n_calls, mode)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return _run_single(base_model, system_prompt, user_prompt, mode)


def _run_single(base_model: str, system_prompt: str, user_prompt: str, mode: str) -> dict:
    """Single Claude call → parsed result."""
    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    try:
        raw_text = _call_claude_cli(full_prompt, model=base_model)
    except ClaudeCLIError as e:
        logger.error(f"Claude CLI failed: {e}")
        return {
            "label": "UNCERTAIN", "confidence": 0.5, "reason": f"cli_error: {str(e)[:100]}",
            "base_model_used": base_model, "mode": mode, "n_calls": 0,
        }

    parsed = _parse_response(raw_text)
    logger.info(f"LLM {mode} ({base_model}): label={parsed['label']}, conf={parsed['confidence']:.2f}")
    return {
        "label": parsed["label"],
        "confidence": parsed["confidence"],
        "reason": parsed.get("reason", ""),
        "base_model_used": base_model,
        "mode": mode,
        "n_calls": 1,
    }


def _run_bagging(base_model: str, system_prompt: str, user_prompt: str, n_calls: int, mode: str) -> dict:
    """Multiple Claude calls → majority vote."""
    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    results = []
    for i in range(n_calls):
        try:
            raw_text = _call_claude_cli(full_prompt, model=base_model)
            parsed = _parse_response(raw_text)
            results.append(parsed)
        except ClaudeCLIError as e:
            logger.warning(f"bagging call {i+1}/{n_calls} failed: {e}")
        # Small pause between bagging calls
        if i < n_calls - 1:
            time.sleep(DEFAULT_PAUSE)

    if not results:
        return {
            "label": "UNCERTAIN", "confidence": 0.5, "reason": "all_bagging_failed",
            "base_model_used": base_model, "mode": "bagging", "n_calls": 0,
        }

    labels = [r["label"] for r in results if r["label"] in ("REAL", "FAKE")]
    if not labels:
        return {
            "label": "UNCERTAIN", "confidence": 0.5, "reason": "all_bagging_uncertain",
            "base_model_used": base_model, "mode": "bagging", "n_calls": len(results),
        }

    counter = Counter(labels)
    winner, winner_count = counter.most_common(1)[0]
    vote_fraction = winner_count / len(labels)
    winner_confs = [r["confidence"] for r in results if r["label"] == winner]
    avg_conf = sum(winner_confs) / len(winner_confs)
    confidence = vote_fraction * avg_conf
    reasons = [r["reason"] for r in results if r["label"] == winner and r["reason"]]

    logger.info(f"LLM bagging ({base_model}): label={winner}, votes={dict(counter)}, conf={confidence:.2f}")
    return {
        "label": winner,
        "confidence": round(confidence, 4),
        "reason": reasons[0] if reasons else "",
        "base_model_used": base_model,
        "mode": "bagging",
        "n_calls": len(results),
        "votes": dict(counter),
    }


# ─────────────────────────────────────────────────────────────────
# PUBLIC API: Batch prediction
# ─────────────────────────────────────────────────────────────────

def predict_batch_with_preset(
    texts: list[str],
    preset_config: dict,
    batch_size: int = 12,
    pause_between: float = 1.0,
) -> list[dict]:
    """
    Batch classification — обробляє СОТНІ текстів через batched Claude calls.

    Об'єднує по `batch_size` текстів у один Claude call → розбирає JSON array.
    Це у ~10x швидше ніж по одному.

    Args:
        texts: список текстів для класифікації
        preset_config: preset configuration (base_model, mode, few_shot_examples)
        batch_size: скільки текстів за один Claude call (default 12)
        pause_between: пауза між batches у секундах (default 1.0)

    Returns:
        list of dicts (один per text у тому ж порядку):
        [{label, confidence, reason, base_model_used, mode}, ...]
    """
    _check_api_key_unset()

    if not texts:
        return []

    base_model = preset_config.get("base_model", DEFAULT_BASE_MODEL)
    mode = preset_config.get("mode", "zero_shot")

    available, error = _check_claude_available()
    if not available:
        logger.error("Claude CLI unavailable for batch: %s", error)
        return [
            {
                "label": "UNCERTAIN",
                "confidence": 0.5,
                "reason": f"claude_cli_unavailable: {error}",
                "base_model_used": "none",
                "mode": mode,
            }
            for _ in texts
        ]

    custom_system = preset_config.get("system_prompt")

    # For batch, prefer batch-specific system prompt
    system_prompt = custom_system or DEFAULT_BATCH_SYSTEM_PROMPT
    examples = preset_config.get("few_shot_examples") if mode == "few_shot" else None

    n_total = len(texts)
    n_batches = (n_total + batch_size - 1) // batch_size

    logger.info(f"Batch eval: {n_total} samples, {n_batches} batches × {batch_size}, model={base_model}")

    all_results = []
    start_time = time.time()

    for batch_idx in range(n_batches):
        batch_start_idx = batch_idx * batch_size
        batch_end_idx = min(batch_start_idx + batch_size, n_total)
        batch_texts = texts[batch_start_idx:batch_end_idx]

        user_prompt = _build_batch_user_prompt(batch_texts, examples=examples)
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        try:
            raw_text = _call_claude_cli(full_prompt, model=base_model, timeout=BATCH_TIMEOUT)
            raw_results = _parse_batch_response(raw_text, expected_count=len(batch_texts))
        except ClaudeCLIError as e:
            logger.error(f"Batch {batch_idx+1}/{n_batches} failed: {e}")
            raw_results = [None] * len(batch_texts)
            cli_error = str(e)[:100]
        else:
            cli_error = None

        # Convert None slots to UNCERTAIN; annotate metadata for backward compat.
        batch_results = []
        for r in raw_results:
            if r is None:
                reason = f"batch_failed: {cli_error}" if cli_error else "missing_in_response"
                r = {"label": "UNCERTAIN", "confidence": 0.5, "reason": reason}
            r["base_model_used"] = base_model
            r["mode"] = mode
            batch_results.append(r)

        all_results.extend(batch_results)

        # Progress
        elapsed = time.time() - start_time
        eta = (n_batches - batch_idx - 1) * (elapsed / (batch_idx + 1))
        logger.info(
            f"Batch {batch_idx+1}/{n_batches} done "
            f"({batch_end_idx}/{n_total} samples, {elapsed:.0f}s elapsed, ETA {eta:.0f}s)"
        )

        # Pause between batches (rate limit safety for Max plan)
        if batch_idx < n_batches - 1:
            time.sleep(pause_between)

    return all_results


# ─────────────────────────────────────────────────────────────────
# LEGACY WRAPPERS
# ─────────────────────────────────────────────────────────────────

def predict(text: str, mode: str = "single", feature_values: dict | None = None) -> dict:
    """Legacy wrapper for /evaluate endpoint and analyze_text LLM branch."""
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