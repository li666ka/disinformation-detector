# api/sources/cross_platform.py
"""
Cross-platform verification service.

Ідея: взяти заголовок або URL новини з RSS-джерела (типово авторитетне) і
перевірити, як соцмережі на неї реагують. Це дає потужні сигнали для fake
news detection:

- Якщо новина є у мейнстрімних RSS (BBC, Reuters, NYT) + обговорюється у Bluesky +
  багато постів мають тон підтвердження → висока довіра
- Якщо новина є тільки в соцмережах (немає у RSS), багато реплаїв з question
  marks, низька ratio верифікованих акаунтів → сигнал можливого фейку
- Якщо заголовок з'являється у Bluesky/Mastodon майже дослівно (копіпаста) без
  посилань на першоджерело → можливе "амбулансове" поширення

Функція verify_news(news_item) повертає:
- original: оригінальна новина (RSS item)
- related_posts: список постів з Bluesky/Mastodon, які обговорюють цю новину
- stats: агреговані показники по related_posts
- signals: список сигналів fake/real
"""
from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter
from typing import Optional
from urllib.parse import urlparse

from .base import NewsItem, SourceError
from . import get_source

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
#  PATCH для api/sources/cross_platform.py
#
#  Замінити функцію extract_keywords_from_title (близько рядка 775)
#  на версію нижче.
# ─────────────────────────────────────────────────────────────────────────

import re

_TITLE_PREFIX_RE = re.compile(
    r"^(BREAKING|UPDATE|UPDATED|EXCLUSIVE|LIVE|WATCH|VIDEO|PHOTOS?|NEW|JUST\s+IN)"
    r"[:\-–—]?\s*",
    flags=re.IGNORECASE,
)

# Latin-script токен; дозволяємо внутрішні дефіси/апострофи,
# щоб "COVID-19", "don't", "state-of-the-art" бралися цілком.
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'’\-]*")

_STOP_WORDS = frozenset({
    # Articles / determiners
    "the", "a", "an", "this", "that", "these", "those", "such", "some", "any",
    "all", "every", "each", "no", "other", "another",
    # Pronouns
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their", "mine", "yours",
    "one", "ones", "who", "whom", "whose",
    # Conjunctions
    "and", "or", "but", "nor", "yet", "so", "if", "than", "because", "while",
    "although", "though", "whereas", "since",
    # Common prepositions
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as", "into",
    "onto", "upon", "about", "against", "between", "through", "during",
    "before", "after", "above", "below", "over", "under", "near", "off",
    "out", "up", "down", "across", "along",
    # Auxiliaries / common verbs
    "is", "was", "are", "were", "be", "been", "being", "am",
    "have", "has", "had", "having",
    "do", "does", "did", "doing", "done",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    # Newsy fillers
    "says", "said", "saying", "told", "tells", "report", "reports", "reported",
    "reportedly", "according", "amid", "amidst", "despite", "plus",
    # Wh-words
    "what", "which", "when", "where", "why", "how",
    # Other frequent low-signal words
    "also", "just", "still", "only", "even", "now", "then", "here", "there",
    "too", "very", "more", "most", "less", "many", "much", "few",
    "get", "gets", "got", "make", "makes", "made", "take", "takes", "took",
    "like", "likely", "new", "old", "year", "years", "today", "yesterday",
})


def extract_keywords_from_title(title: str, max_keywords: int = 4) -> str:
    """
    Витягнути найбільш інформативні слова з заголовку (Latin-only).

    Стратегія:
      1. Прибираємо новинні префікси (BREAKING:, JUST IN, LIVE, ...).
      2. Токенізуємо; дозволяємо внутрішні дефіси/апострофи
         ("COVID-19", "don't", "state-of-the-art").
      3. Відкидаємо стоп-слова, надто короткі токени.
      4. Пріоритет:
         - абревіатури (UN, NATO, FBI, NASA) — всі літери великі, 2–6 символів;
         - власні імена / title-case (Biden, Ukraine);
         - решта слів довжиною ≥ 5 літер.
      5. Повертаємо до `max_keywords` слів, зберігаючи порядок появи.

    Чому max_keywords=4, а не 5:
      Bluesky і Mastodon трактують пробіли у запиті як AND. 5 специфічних
      слів часто дає 0 результатів; 3-4 — золота середина для recall.

    Приклади:
      "Breaking: Ukraine announces new defense package from allies"
        → "Ukraine announces defense package"
      "NATO members to increase defense spending by 2026"
        → "NATO members increase defense"
      "New COVID-19 variant detected in UK"
        → "COVID-19 UK variant detected"
    """
    if not title:
        return ""

    title = _TITLE_PREFIX_RE.sub("", title)
    tokens = _WORD_RE.findall(title)

    acronyms: list[str] = []
    proper: list[str] = []
    content: list[str] = []
    seen: set[str] = set()

    for w in tokens:
        lw = w.lower()
        if lw in _STOP_WORDS or lw in seen:
            continue
        if len(w) < 3:
            continue
        seen.add(lw)

        # Акронім: короткий токен, всі літери великі (UN, NATO, SCOTUS)
        if w.isupper() and 2 <= len(w) <= 6:
            acronyms.append(w)
        # Власна назва / title-case (перша велика, решта — ні)
        elif w[0].isupper() and not w.isupper():
            proper.append(w)
        elif len(w) >= 5:
            content.append(w)

    # Адаптивний вибір:
    #   - якщо є ≥2 named entities (акроніми + TitleCase) — повертаємо їх (precision-orient):
    #     2-3 entity-токени + 1 content-слово якщо є.
    #   - інакше — звичайний top-N (recall-orient), щоб не повертати порожнє коли
    #     заголовок узагалі без власних назв ("vaccine causes autism").
    entities = acronyms + proper
    if len(entities) >= 2:
        result = entities[:3]
        if content:
            result.append(content[0])
    else:
        result = (acronyms + proper + content)[:max_keywords]
    return " ".join(result)

def extract_domain(url: str) -> Optional[str]:
    """example.com з https://example.com/path/article"""
    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host if host else None
    except Exception:
        return None


async def verify_news(
    news_item: NewsItem,
    social_sources: list[str] = None,
    limit_per_source: int = 10,
) -> dict:
    """
    Шукати пости у соцмережах, які обговорюють новину з RSS.

    Args:
      news_item: оригінальна новина (зазвичай з RSS, але підходить будь-яке джерело)
      social_sources: ["bluesky", "mastodon"] за замовчуванням
      limit_per_source: скільки постів брати з кожної платформи

    Returns:
      {
        "original": NewsItem dict,
        "query_used": "keywords used for search",
        "related_posts": [post1, post2, ...],  # з соцмереж
        "stats": {
          "total_related": N,
          "by_source": {"bluesky": N1, "mastodon": N2},
          "total_engagement": {...},
          "verified_authors_pct": 0.3,
          "avg_account_age_days": 1250,
          "domain_mentioned_count": 5,  # скільки постів згадують домен першоджерела
        },
        "signals": [
          {"type": "low_verified_ratio", "value": 0.1, "note": "..."},
          ...
        ]
      }
    """
    if social_sources is None:
        social_sources = ["bluesky", "mastodon"]

    # Build search query
    title = news_item.title or news_item.text[:200]
    query = extract_keywords_from_title(title)
    if not query:
        # Fallback: перші 5 слів з тексту
        query = " ".join(news_item.text.split()[:5])

    logger.info(f"verify_news: searching '{query}' on {social_sources}")

    # Fetch posts concurrently
    async def _search_one(src_name: str):
        try:
            src = get_source(src_name)
            return src_name, await src.search(query, limit=limit_per_source)
        except SourceError as e:
            logger.warning(f"verify_news: {src_name} failed: {e}")
            return src_name, []
        except Exception as e:
            logger.warning(f"verify_news: {src_name} unexpected: {e}")
            return src_name, []

    results = await asyncio.gather(
        *(_search_one(s) for s in social_sources),
        return_exceptions=False,
    )

    # Flatten
    all_posts: list[NewsItem] = []
    by_source: dict[str, list[NewsItem]] = {}
    for src_name, posts in results:
        by_source[src_name] = posts
        all_posts.extend(posts)

    # ── Compute stats ─────────────────────────────────────────────────────
    stats = _compute_stats(news_item, all_posts, by_source)
    signals = _detect_signals(stats, all_posts)

    return {
        "original": news_item.to_dict(),
        "query_used": query,
        "related_posts": [p.to_dict() for p in all_posts],
        "stats": stats,
        "signals": signals,
    }


def _compute_stats(
    original: NewsItem,
    all_posts: list[NewsItem],
    by_source: dict[str, list[NewsItem]],
) -> dict:
    """Aggregate statistics across related posts."""
    total = len(all_posts)

    # By source counts
    source_counts = {src: len(posts) for src, posts in by_source.items()}

    # Engagement aggregates
    total_likes = sum((p.likes_count or 0) for p in all_posts)
    total_reposts = sum((p.reposts_count or 0) for p in all_posts)
    total_replies = sum((p.replies_count or 0) for p in all_posts)

    # Verified authors ratio
    posts_with_verified_info = [p for p in all_posts if p.author_is_verified is not None]
    verified_count = sum(1 for p in posts_with_verified_info if p.author_is_verified)
    verified_ratio = (
        verified_count / len(posts_with_verified_info)
        if posts_with_verified_info
        else None
    )

    # Custom domain ratio (Bluesky-specific signal of trusted accounts)
    posts_with_cd_info = [p for p in all_posts if p.author_has_custom_domain is not None]
    cd_count = sum(1 for p in posts_with_cd_info if p.author_has_custom_domain)
    custom_domain_ratio = (
        cd_count / len(posts_with_cd_info)
        if posts_with_cd_info
        else None
    )

    # Account age
    ages = [p.author_account_age_days for p in all_posts if p.author_account_age_days is not None]
    avg_age = sum(ages) / len(ages) if ages else None
    min_age = min(ages) if ages else None

    # Followers
    followers = [p.author_followers_count for p in all_posts if p.author_followers_count is not None]
    avg_followers = sum(followers) / len(followers) if followers else None
    median_followers = sorted(followers)[len(followers) // 2] if followers else None

    # Domain matching — how many posts mention the original source's domain
    original_domain = extract_domain(original.url)
    domain_mentions = 0
    if original_domain:
        for p in all_posts:
            if original_domain in p.text.lower():
                domain_mentions += 1

    # URL presence
    with_url = sum(1 for p in all_posts if p.has_url_in_text)
    url_ratio = with_url / total if total else 0

    # Reply ratio (are these organic posts or replies?)
    reply_count = sum(1 for p in all_posts if p.is_reply)
    reply_ratio = reply_count / total if total else 0

    return {
        "total_related": total,
        "by_source": source_counts,
        "original_domain": original_domain,
        "domain_mentioned_count": domain_mentions,
        "total_engagement": {
            "likes": total_likes,
            "reposts": total_reposts,
            "replies": total_replies,
        },
        "verified_authors_pct": round(verified_ratio * 100, 1) if verified_ratio is not None else None,
        "custom_domain_authors_pct": round(custom_domain_ratio * 100, 1) if custom_domain_ratio is not None else None,
        "avg_account_age_days": round(avg_age) if avg_age is not None else None,
        "min_account_age_days": min_age,
        "avg_followers_count": round(avg_followers) if avg_followers is not None else None,
        "median_followers_count": median_followers,
        "posts_with_url_pct": round(url_ratio * 100, 1),
        "reply_ratio_pct": round(reply_ratio * 100, 1),
    }


def _detect_signals(stats: dict, posts: list[NewsItem]) -> list[dict]:
    """
    Виявити підозрілі або довірчі сигнали на основі агрегатів.
    Кожен сигнал має type (machine-readable), severity (info|warn|alert),
    та note (людяний опис).
    """
    signals = []
    total = stats.get("total_related", 0)

    if total == 0:
        signals.append({
            "type": "no_social_discussion",
            "severity": "warn",
            "note": "Новина не обговорюється у соцмережах — незвично для важливої події",
        })
        return signals

    # Low verified ratio
    verified_pct = stats.get("verified_authors_pct")
    if verified_pct is not None and verified_pct < 10 and total >= 5:
        signals.append({
            "type": "low_verified_ratio",
            "severity": "warn",
            "value": verified_pct,
            "note": f"Лише {verified_pct}% авторів мають верифікацію",
        })

    # Many very young accounts (< 30 days)
    young_posts = [p for p in posts if p.author_account_age_days is not None and p.author_account_age_days < 30]
    if len(young_posts) / total >= 0.3 and total >= 5:
        signals.append({
            "type": "many_young_accounts",
            "severity": "alert",
            "value": len(young_posts),
            "note": f"{len(young_posts)} з {total} постів від акаунтів молодше 30 днів (можливі боти)",
        })

    # Many low-follower accounts
    low_follower_posts = [
        p for p in posts
        if p.author_followers_count is not None and p.author_followers_count < 50
    ]
    if total >= 5 and len(low_follower_posts) / total >= 0.5:
        signals.append({
            "type": "many_low_follower_accounts",
            "severity": "warn",
            "value": len(low_follower_posts),
            "note": f"Половина або більше авторів мають <50 фоловерів",
        })

    # Domain mentions (link back to source)
    domain_mentions = stats.get("domain_mentioned_count", 0)
    if stats.get("original_domain") and total >= 5:
        if domain_mentions == 0:
            signals.append({
                "type": "no_source_link",
                "severity": "warn",
                "note": "Жоден пост не згадує домен оригінального джерела — новина обговорюється без посилань",
            })
        elif domain_mentions / total >= 0.3:
            signals.append({
                "type": "high_source_attribution",
                "severity": "info",
                "value": domain_mentions,
                "note": f"{domain_mentions} з {total} постів згадують оригінальне джерело",
            })

    # High engagement without source verification
    total_engagement = (
        stats["total_engagement"]["likes"]
        + stats["total_engagement"]["reposts"]
        + stats["total_engagement"]["replies"]
    )
    if total_engagement > 10000 and domain_mentions == 0:
        signals.append({
            "type": "viral_without_source",
            "severity": "alert",
            "value": total_engagement,
            "note": f"Висока залученість ({total_engagement}) без посилань на першоджерело — віральний контент з неперевірених акаунтів",
        })

    # Multiple platforms discussing (positive signal)
    by_source = stats.get("by_source", {})
    active_sources = [s for s, n in by_source.items() if n > 0]
    if len(active_sources) >= 2:
        signals.append({
            "type": "multi_platform_discussion",
            "severity": "info",
            "value": len(active_sources),
            "note": f"Новина обговорюється на {len(active_sources)} платформах одночасно",
        })

    return signals