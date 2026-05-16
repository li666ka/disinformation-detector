"""LLM extraction структурованих даних із соціального поста (Mastodon/Bluesky).

Окремий від api.claim_extractor (той — для fact_check.verify_post, з іншим
контрактом). Цей модуль обслуговує `routers/real_world.py`.

Витягає:
  - is_news_claim + claim_text/subject/type
  - stance автора, emotion, confidence
  - cited_sources, has_url

Використовує спільну Claude CLI обгортку з `llm_predictor` (Max-plan OAuth).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from api.llm_predictor import (
    BASE_MODELS,
    DEFAULT_BASE_MODEL,
    ClaudeCLIError,
    _call_claude_cli,
    _check_environment,
)
from api.social_fetchers import SocialPost

log = logging.getLogger(__name__)

EXTRACTION_TIMEOUT = 90


EXTRACTION_PROMPT = """Ти експерт з аналізу соціальних мереж та фактчекінгу.

Тобі дано пост з соціальної мережі. Твоя задача — розпакувати з нього структурований JSON.

ПОСТ:
Автор: {author}
Платформа: {platform}
Мова: {language}
Текст:
\"\"\"
{content}
\"\"\"

Поверни ТІЛЬКИ JSON у такому форматі, без жодного додаткового тексту:

{{
  "is_news_claim": true|false,
  "reasoning_for_is_news": "коротке пояснення чому це новина або ні",

  "claim_text": "перефразована новина в одному реченні. Тільки якщо is_news_claim=true.",
  "claim_subject": "хто/що в центрі новини",
  "claim_type": "politics|celebrity|science|health|economy|sport|technology|society|other",

  "author_stance": "supporting|skeptical|neutral|ironic|outraged|amused|sympathetic|unclear",
  "stance_reasoning": "коротке пояснення позиції автора",

  "emotion": "anger|fear|joy|sadness|surprise|disgust|trust|anticipation|neutral",
  "emotion_intensity": "low|medium|high",

  "confidence_in_claim": "high|medium|low",
  "confidence_reasoning": "наскільки впевнено автор подає інформацію",

  "cited_sources": ["назви джерел якщо є, наприклад 'Page Six', 'BBC'"],
  "has_url": true|false,

  "language_detected": "en|uk|ru|...",
  "is_translation": true|false
}}

ВАЖЛИВІ ПРАВИЛА:
1. Якщо пост — це особиста думка/досвід без verifiable claim → is_news_claim=false
2. Якщо пост — це жарт/мем без серйозного твердження → is_news_claim=false
3. Якщо пост ЦИТУЄ або ПЕРЕКАЗУЄ новину (навіть з коментарем) → is_news_claim=true
4. claim_text має бути ОБ'ЄКТИВНОЮ перефразовкою без емоцій автора
5. Поверни ТІЛЬКИ JSON. Без preamble, без markdown, без backticks.
"""


CLASSIFICATION_PROMPT = """Ти експерт-фактчекер. Класифікуй наступне твердження як TRUE (правда), FALSE (фейк) або UNCERTAIN (недостатньо інформації).

Твердження: "{claim}"

Поверни ТІЛЬКИ JSON:
{{
  "verdict": "TRUE|FALSE|UNCERTAIN",
  "confidence": 0.0-1.0,
  "reasoning": "детальне пояснення в 2-3 реченнях чому ти прийшов до цього висновку",
  "would_need_to_verify": ["що треба перевірити для впевненості"]
}}

Без markdown, без preamble.
"""


@dataclass
class ExtractedClaim:
    is_news_claim: bool
    reasoning_for_is_news: str

    claim_text: Optional[str]
    claim_subject: Optional[str]
    claim_type: Optional[str]

    author_stance: str
    stance_reasoning: str

    emotion: str
    emotion_intensity: str

    confidence_in_claim: str
    confidence_reasoning: str

    cited_sources: list[str]
    has_url: bool

    language_detected: Optional[str]
    is_translation: bool

    raw_llm_output: Optional[str] = None


def _resolve_model(alias: str) -> str:
    """Map UI alias (claude-haiku) → офіційний model id (claude-haiku-4-5)."""
    if alias in BASE_MODELS:
        return alias
    short = (alias or "").lower()
    mapping = {
        "claude-haiku": "claude-haiku-4-5",
        "haiku": "claude-haiku-4-5",
        "claude-sonnet": "claude-sonnet-4-6",
        "sonnet": "claude-sonnet-4-6",
        "claude-opus": "claude-opus-4-7",
        "opus": "claude-opus-4-7",
    }
    return mapping.get(short, DEFAULT_BASE_MODEL)


def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.+?)\s*```", t, re.DOTALL)
    return m.group(1).strip() if m else t


def _extract_json_object(text: str) -> dict:
    cleaned = _strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.+\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError("no JSON object found in LLM output")
    return json.loads(match.group(0))


def extract_post(post: SocialPost, model: str = DEFAULT_BASE_MODEL) -> ExtractedClaim:
    """Витягти структуровану інформацію з поста через Claude CLI."""
    _check_environment()
    resolved_model = _resolve_model(model)

    prompt = EXTRACTION_PROMPT.format(
        author=post.author_handle,
        platform=post.platform,
        language=post.language or "unknown",
        content=post.content,
    )

    log.info(f"Extracting post {post.post_id} via {resolved_model}")

    try:
        raw_output = _call_claude_cli(prompt, model=resolved_model, timeout=EXTRACTION_TIMEOUT)
    except ClaudeCLIError as e:
        raise RuntimeError(f"Claude CLI error: {e}") from e

    try:
        data = _extract_json_object(raw_output)
    except (ValueError, json.JSONDecodeError) as e:
        log.error(f"Failed to parse extraction output: {e}\nRaw: {raw_output[:500]}")
        raise RuntimeError(f"LLM returned invalid JSON: {e}") from e

    sources = data.get("cited_sources")
    if not isinstance(sources, list):
        sources = []

    return ExtractedClaim(
        is_news_claim=bool(data.get("is_news_claim", False)),
        reasoning_for_is_news=str(data.get("reasoning_for_is_news") or ""),
        claim_text=data.get("claim_text") or None,
        claim_subject=data.get("claim_subject") or None,
        claim_type=data.get("claim_type") or None,
        author_stance=str(data.get("author_stance") or "unclear"),
        stance_reasoning=str(data.get("stance_reasoning") or ""),
        emotion=str(data.get("emotion") or "neutral"),
        emotion_intensity=str(data.get("emotion_intensity") or "low"),
        confidence_in_claim=str(data.get("confidence_in_claim") or "medium"),
        confidence_reasoning=str(data.get("confidence_reasoning") or ""),
        cited_sources=[str(s) for s in sources],
        has_url=bool(data.get("has_url", False)),
        language_detected=data.get("language_detected") or None,
        is_translation=bool(data.get("is_translation", False)),
        raw_llm_output=raw_output,
    )


def classify_claim(claim_text: str, model: str = DEFAULT_BASE_MODEL) -> dict:
    """LLM verdict для одного claim. Повертає {verdict, confidence, reasoning, would_need_to_verify, model}."""
    _check_environment()
    resolved_model = _resolve_model(model)
    prompt = CLASSIFICATION_PROMPT.format(claim=claim_text)

    try:
        raw_output = _call_claude_cli(prompt, model=resolved_model, timeout=EXTRACTION_TIMEOUT)
    except ClaudeCLIError as e:
        raise RuntimeError(f"Claude CLI error: {e}") from e

    try:
        data = _extract_json_object(raw_output)
    except (ValueError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Classification returned invalid JSON: {e}") from e

    verdict = str(data.get("verdict") or "UNCERTAIN").upper()
    if verdict not in {"TRUE", "FALSE", "UNCERTAIN"}:
        verdict = "UNCERTAIN"

    try:
        confidence = float(data.get("confidence") or 0.5)
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    needs_verify = data.get("would_need_to_verify")
    if not isinstance(needs_verify, list):
        needs_verify = []

    return {
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": str(data.get("reasoning") or "").strip(),
        "would_need_to_verify": [str(x) for x in needs_verify],
        "model": resolved_model,
    }
