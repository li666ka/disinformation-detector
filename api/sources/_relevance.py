"""Лексична relevance-фільтрація після fetch.

Bluesky/Mastodon search дають AND по словах і часто повертають тематично
далеких сусідів (особливо коли query — 1 акронім або 2 слова). Цей модуль
скорить кожен NewsItem за token-overlap з query і дозволяє ранжувати /
фільтрувати на рівні роутера. Джерела (BaseNewsSource.search) залишаються
без змін.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

_STOP: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "of", "to", "in",
    "on", "at", "for", "with", "by", "from", "and", "or", "but", "not",
    "this", "that", "it", "its", "do", "does", "did", "has", "have", "had",
    "will", "would", "can", "could", "may", "might", "should", "said", "says",
})

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z'-]{2,}")


def _tokenize(text: str) -> set[str]:
    """Lowercase tokens >=3 chars без стоп-слів."""
    if not text:
        return set()
    return {t for t in (m.group(0).lower() for m in _TOKEN_RE.finditer(text)) if t not in _STOP}


def relevance_score(
    query: str,
    post_text: str,
    post_title: Optional[str] = None,
) -> float:
    """Частка query-keywords присутніх у пості (∈ [0, 1]).

    Інтерпретація порогів:
      ≥0.5 — точний / релевантний збіг
      ≥0.3 — слабка релевантність (можливо тематично пов'язано)
      <0.3 — навряд про те саме
    """
    if not query or not post_text:
        return 0.0
    q_tokens = _tokenize(query)
    if not q_tokens:
        return 0.0
    full_text = ((post_title or "") + " " + post_text).strip()
    p_tokens = _tokenize(full_text)
    if not p_tokens:
        return 0.0
    return len(q_tokens & p_tokens) / len(q_tokens)


def filter_by_relevance(
    query: str,
    items: Iterable,
    min_score: float = 0.3,
) -> list:
    """Залишити items зі score ≥ `min_score`, відсортовані спаданням."""
    scored: list[tuple[float, object]] = []
    for item in items:
        text = getattr(item, "text", "") or ""
        title = getattr(item, "title", None)
        score = relevance_score(query, text, title)
        if score >= min_score:
            scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored]
