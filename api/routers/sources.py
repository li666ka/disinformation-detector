# api/routers/sources.py
"""
Router для джерел даних: пошук, свіжі пости, fetch за URL.

Endpoints:
- GET  /sources/search?query=...&sources=bluesky,mastodon&limit=20
- GET  /sources/recent?sources=bluesky,rss&limit=20
- POST /sources/fetch-url  { "url": "https://bsky.app/..." }
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, HttpUrl

from api.auth import get_current_user
from api.database import User
from api.sources import (
    BaseNewsSource,
    NewsItem,
    SourceError,
    get_all_sources,
    get_source,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sources", tags=["sources"])


# ── Request/Response schemas ──────────────────────────────────────────────

class SourceSearchResponse(BaseModel):
    posts: list[dict]
    total: int
    sources_used: list[str]


class FetchUrlRequest(BaseModel):
    url: str  # не HttpUrl — щоб не падати при неочікуваних схемах


class PostDetailsResponse(BaseModel):
    post: dict
    replies: list[dict]
    reposted_by: list[dict]
    liked_by: list[dict]
    quoted_by: list[dict]
    stats: Optional[dict] = None
    fetched_limits: Optional[dict] = None

# ── Helpers ───────────────────────────────────────────────────────────────

def _parse_sources_param(sources_csv: str) -> list[str]:
    """Розпарсити "bluesky,mastodon,rss" → ["bluesky", "mastodon", "rss"]."""
    valid = {"bluesky", "mastodon", "rss"}
    requested = [s.strip().lower() for s in sources_csv.split(",") if s.strip()]
    filtered = [s for s in requested if s in valid]

    if not filtered:
        raise HTTPException(
            status_code=400,
            detail=f"Не вказано жодного валідного джерела. Доступні: {sorted(valid)}",
        )
    return filtered


async def _run_across_sources(
    source_names: list[str],
    action: str,  # "search" | "recent"
    **kwargs,
) -> tuple[list[NewsItem], list[str], list[str]]:
    """
    Паралельно запускає action на всіх обраних джерелах.
    Повертає (items, successful_sources, failed_sources_with_errors).
    """
    sources = [get_source(name) for name in source_names]

    async def _call(src: BaseNewsSource):
        if action == "search":
            return await src.search(kwargs["query"], kwargs["limit"])
        elif action == "recent":
            return await src.get_recent(kwargs["limit"])
        else:
            raise ValueError(f"Unknown action: {action}")

    tasks = [_call(src) for src in sources]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    items: list[NewsItem] = []
    successful: list[str] = []
    errors: list[str] = []

    for src, result in zip(sources, results):
        if isinstance(result, SourceError):
            errors.append(f"{src.source_name}: {result.message}")
            logger.warning(f"Source '{src.source_name}' failed: {result}")
        elif isinstance(result, Exception):
            errors.append(f"{src.source_name}: {str(result)[:100]}")
            logger.error(f"Source '{src.source_name}' unexpected: {result}", exc_info=True)
        else:
            items.extend(result)
            successful.append(src.source_name)

    return items, successful, errors


def _items_to_dicts(items: list[NewsItem], limit: int) -> list[dict]:
    """Конвертувати NewsItem → dict та обрізати до ліміту."""
    # Якщо з кількох джерел — чергуємо, щоб не було багато Bluesky зверху,
    # потім багато Mastodon тощо. Простий round-robin:
    by_source: dict[str, list[NewsItem]] = {}
    for item in items:
        by_source.setdefault(item.source, []).append(item)

    interleaved: list[NewsItem] = []
    max_per_source = max((len(v) for v in by_source.values()), default=0)
    for i in range(max_per_source):
        for src_name in by_source:
            if i < len(by_source[src_name]):
                interleaved.append(by_source[src_name][i])

    return [item.to_dict() for item in interleaved[:limit]]


# ── Endpoints ─────────────────────────────────────────────────────────────

@router.get("/search", response_model=SourceSearchResponse)
async def search_posts(
    query: str = Query(..., min_length=1, max_length=200),
    sources: str = Query(..., description="Comma-separated: bluesky,mastodon,rss"),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
):
    """Пошук постів за ключовими словами у обраних джерелах."""
    source_names = _parse_sources_param(sources)

    items, successful, errors = await _run_across_sources(
        source_names, "search", query=query, limit=limit,
    )

    if not successful:
        # Жодне джерело не спрацювало — це помилка
        raise HTTPException(
            status_code=502,
            detail="Усі джерела недоступні: " + "; ".join(errors),
        )

    return SourceSearchResponse(
        posts=_items_to_dicts(items, limit),
        total=len(items),
        sources_used=successful,
    )


@router.get("/recent", response_model=SourceSearchResponse)
async def get_recent_posts(
    sources: str = Query(..., description="Comma-separated: bluesky,mastodon,rss"),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
):
    """Свіжі (останні) пости з обраних джерел."""
    source_names = _parse_sources_param(sources)

    items, successful, errors = await _run_across_sources(
        source_names, "recent", limit=limit,
    )

    if not successful:
        raise HTTPException(
            status_code=502,
            detail="Усі джерела недоступні: " + "; ".join(errors),
        )

    return SourceSearchResponse(
        posts=_items_to_dicts(items, limit),
        total=len(items),
        sources_used=successful,
    )


@router.post("/fetch-url")
async def fetch_by_url(
    req: FetchUrlRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Завантажити конкретний пост/статтю за URL.
    Автоматично визначає потрібне джерело за URL pattern.
    """
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL не може бути порожнім")

    all_sources = get_all_sources()

    # Пріоритет: Bluesky → Mastodon → RSS (RSS як fallback для будь-якого URL)
    # Це важливо, бо RSS.can_handle_url() повертає True для будь-якого HTTP URL.
    ordered_sources = sorted(
        all_sources,
        key=lambda s: 0 if s.source_name != "rss" else 1,
    )

    for src in ordered_sources:
        if not src.can_handle_url(url):
            continue
        try:
            item = await src.fetch_by_url(url)
        except SourceError as e:
            logger.warning(f"fetch_by_url: {src.source_name} failed: {e}")
            # Якщо конкретне джерело впевнено належить URL-у, але впало —
            # повертаємо 502. Для RSS fallback — продовжуємо перебір.
            if src.source_name != "rss":
                raise HTTPException(status_code=502, detail=str(e.message)) from e
            continue
        if item is not None:
            return item.to_dict()

    raise HTTPException(
        status_code=404,
        detail="Не вдалося знайти джерело для цього URL або пост відсутній",
    )


# ── GET /sources/post-details ────────────────────────────────────────────

@router.get("/post-details", response_model=PostDetailsResponse)
async def get_post_details_endpoint(
    post_id: str = Query(..., description='Post ID like "bluesky:at://..." or "mastodon:12345" or a URL'),
    max_replies: int = Query(50, ge=1, le=200),
    max_likers: int = Query(100, ge=1, le=300),
    max_reposters: int = Query(100, ge=1, le=300),
    _: User = Depends(get_current_user),
):
    """
    Return full details for a single post:
    - replies (with author profiles)
    - reposted_by (profiles of users who reposted)
    - liked_by (profiles of users who liked)
    - quoted_by (Bluesky only; Mastodon returns empty)
    - stats (aggregated analysis of all participants)
    - fetched_limits (what was actually fetched vs total)

    Accepts multiple ID formats:
      - "bluesky:at://did:plc:.../app.bsky.feed.post/rkey"
      - "at://did:plc:.../app.bsky.feed.post/rkey"
      - "https://bsky.app/profile/handle/post/rkey"
      - "mastodon:12345"
      - "12345" (Mastodon numeric ID)
      - "https://mastodon.social/@user/12345"
    """
    # Try each source until one handles this post_id
    details = None
    last_err = None

    for src in get_all_sources():
        try:
            result = await src.get_post_details(
                post_id,
                max_replies=max_replies,
                max_likers=max_likers,
                max_reposters=max_reposters,
            )
            if result:
                details = result
                break
        except SourceError as e:
            last_err = e
            logger.warning(f"{src.source_name} post-details failed: {e}")
            continue
        except Exception as e:
            last_err = e
            logger.warning(f"{src.source_name} post-details unexpected: {e}")
            continue

    if not details:
        msg = f"Could not fetch details for post_id={post_id}"
        if last_err:
            msg += f" (last error: {last_err})"
        raise HTTPException(status_code=404, detail=msg)

    return PostDetailsResponse(**details.to_dict())