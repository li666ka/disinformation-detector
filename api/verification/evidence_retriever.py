# api/verification/evidence_retriever.py
"""
Evidence Retrieval — шукає evidence для claim у RSS + соц-мережах.

Стратегія:
1. Build search query з claim (key entities + main verb phrase)
2. Паралельно виконати пошук через наявні BlueskyClient, MastodonClient,
   RSS feeds (via sources module).
3. Конвертувати результати у Evidence об'єкти.
4. Обчислити authority_weight для кожного evidence.

НЕ виконує stance detection — це окремий крок у stance_detector.py.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from .models import Claim, Evidence, EvidenceBundle

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Authority weighting
# ──────────────────────────────────────────────────────────────────────────

# Відомі авторитетні news sources — вага 1.0
AUTHORITATIVE_DOMAINS = {
    "bbc.com", "bbc.co.uk", "reuters.com", "apnews.com", "nytimes.com",
    "washingtonpost.com", "theguardian.com", "ft.com", "wsj.com",
    "economist.com", "nature.com", "science.org", "nih.gov", "who.int",
    "cdc.gov", "npr.org", "pbs.org", "politifact.com", "snopes.com",
    "factcheck.org", "fullfact.org",
}

# Fact-checker domains (вища вага)
FACT_CHECKER_DOMAINS = {
    "politifact.com", "snopes.com", "factcheck.org", "fullfact.org",
    "reutersagency.com/en/fact-check", "apnews.com/hub/ap-fact-check",
}


def compute_rss_authority(url: Optional[str], source_name: Optional[str]) -> float:
    """
    Вага для RSS-стaтті від 0.0 до 1.0.
    - Fact-checker: 1.0
    - Authoritative news: 0.8
    - Інші: 0.5
    """
    if not url and not source_name:
        return 0.5

    text_to_check = (url or "") + " " + (source_name or "")
    text_lower = text_to_check.lower()

    for domain in FACT_CHECKER_DOMAINS:
        if domain in text_lower:
            return 1.0

    for domain in AUTHORITATIVE_DOMAINS:
        if domain in text_lower:
            return 0.8

    return 0.5


def compute_social_authority(
    verified: Optional[bool],
    followers_count: Optional[int],
    account_age_days: Optional[int],
    has_custom_domain: Optional[bool] = None,
) -> float:
    """
    Вага для соц-поста від 0.0 до 1.0.

    Формула:
    - Верифікований акаунт → +0.4
    - Custom domain (Bluesky handle типу nytimes.com) → +0.3
    - Followers >10000 → +0.2
    - Account age >365 днів → +0.1

    Cap на 1.0.
    """
    weight = 0.1  # base

    if verified is True:
        weight += 0.4
    if has_custom_domain is True:
        weight += 0.3

    if followers_count is not None:
        if followers_count > 100_000:
            weight += 0.3
        elif followers_count > 10_000:
            weight += 0.2
        elif followers_count > 1_000:
            weight += 0.1

    if account_age_days is not None and account_age_days > 365:
        weight += 0.1

    return min(weight, 1.0)


# ──────────────────────────────────────────────────────────────────────────
# Query construction
# ──────────────────────────────────────────────────────────────────────────

def build_search_query(claim: Claim, max_length: int = 100) -> str:
    """
    Побудувати search query з claim.

    Стратегія:
    - Пріоритет entities (named entities дають precision)
    - Додати ключові слова з самого claim.text
    - Обрізати до max_length
    """
    # Якщо entities є — використовуємо їх
    if claim.entities:
        query_parts = list(claim.entities)
        # Додамо кілька ключових слів з тексту (перші non-stop)
        text_words = _extract_keywords(claim.text)
        for word in text_words:
            if word.lower() not in [e.lower() for e in query_parts]:
                query_parts.append(word)
            if len(" ".join(query_parts)) > max_length:
                break
        return " ".join(query_parts)[:max_length]

    # Fallback — key words з тексту
    return " ".join(_extract_keywords(claim.text)[:5])


import re
_STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "must", "can", "this",
    "that", "these", "those", "it", "its", "they", "their", "them",
}


def _extract_keywords(text: str) -> list[str]:
    """Витягти key words з тексту (non-stop, length > 3)."""
    words = re.findall(r"\b[A-Za-z][\w']{2,}\b", text)
    seen = set()
    result = []
    # Пріоритет капіталізованим (proper nouns)
    for w in words:
        if w.lower() in _STOP_WORDS:
            continue
        wl = w.lower()
        if wl in seen:
            continue
        seen.add(wl)
        if w[0].isupper():
            result.append(w)
    for w in words:
        wl = w.lower()
        if wl in _STOP_WORDS or wl in seen:
            continue
        seen.add(wl)
        if len(w) > 4:
            result.append(w)
    return result


# ──────────────────────────────────────────────────────────────────────────
# Main retrieval function
# ──────────────────────────────────────────────────────────────────────────

async def gather_evidence(
    claim: Claim,
    *,
    limit_rss: int = 5,
    limit_social: int = 10,
    sources: Optional[list[str]] = None,
) -> EvidenceBundle:
    """
    Зібрати evidence для claim паралельно з RSS + соц-мереж.

    Args:
        claim: Claim object
        limit_rss: max RSS-статей
        limit_social: max соц-постів на платформу
        sources: список джерел або None → ["rss", "bluesky", "mastodon"]

    Returns:
        EvidenceBundle
    """
    start = time.time()

    if sources is None:
        sources = ["rss", "bluesky", "mastodon"]

    query = build_search_query(claim)
    logger.info(f"gather_evidence: query='{query}', sources={sources}")

    try:
        from api.sources import get_source, SourceError
    except ImportError as e:
        logger.error(f"Cannot import sources module: {e}")
        return EvidenceBundle(claim=claim, query_used=query)

    # Паралельний збір
    async def search_one(src_name: str, limit: int):
        try:
            src = get_source(src_name)
            items = await src.search(query, limit=limit)
            return src_name, items
        except Exception as e:
            logger.warning(f"Search failed for {src_name}: {e}")
            return src_name, []

    tasks = []
    for src_name in sources:
        limit = limit_rss if src_name == "rss" else limit_social
        tasks.append(search_one(src_name, limit))

    results = await asyncio.gather(*tasks, return_exceptions=False)

    # Конвертувати у Evidence
    rss_evidence: list[Evidence] = []
    social_evidence: list[Evidence] = []

    for src_name, items in results:
        for item in items:
            ev = _news_item_to_evidence(item, src_name)
            if ev is None:
                continue
            if src_name == "rss":
                rss_evidence.append(ev)
            else:
                social_evidence.append(ev)

    elapsed = time.time() - start
    total = len(rss_evidence) + len(social_evidence)

    return EvidenceBundle(
        claim=claim,
        rss_evidence=rss_evidence,
        social_evidence=social_evidence,
        total_found=total,
        retrieval_time_seconds=round(elapsed, 2),
        query_used=query,
    )


def _news_item_to_evidence(item, source_type: str) -> Optional[Evidence]:
    """Конвертувати NewsItem з наявного sources module у Evidence."""
    try:
        # item це NewsItem dataclass (або подібний) — дістаємо атрибути
        text = getattr(item, "text", "") or ""
        if not text.strip():
            return None

        title = getattr(item, "title", None)
        url = getattr(item, "url", None)
        published = getattr(item, "created_at", None)
        author = getattr(item, "author", None) or getattr(item, "author_handle", None)

        verified = getattr(item, "author_is_verified", None)
        followers = getattr(item, "author_followers_count", None)
        age = getattr(item, "author_account_age_days", None)
        custom_domain = getattr(item, "author_has_custom_domain", None)

        # Визначаємо source_name
        if source_type == "rss":
            # Для RSS: використовуємо доменну назву як source_name
            source_name = _extract_domain(url) or "RSS"
            authority = compute_rss_authority(url, source_name)
        else:
            source_name = author or source_type
            authority = compute_social_authority(
                verified=verified,
                followers_count=followers,
                account_age_days=age,
                has_custom_domain=custom_domain,
            )

        # Обрізаємо text до розумного розміру для LLM
        text_truncated = text[:800] if len(text) > 800 else text

        return Evidence(
            source_type=source_type,  # type: ignore
            source_name=source_name,
            title=title,
            text=text_truncated,
            url=url,
            published_at=published,
            author=author,
            author_verified=verified,
            author_followers_count=followers,
            authority_weight=authority,
        )
    except Exception as e:
        logger.warning(f"Failed to convert NewsItem to Evidence: {e}")
        return None


def _extract_domain(url: Optional[str]) -> Optional[str]:
    """example.com з https://example.com/path"""
    if not url:
        return None
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host or None
    except Exception:
        return None