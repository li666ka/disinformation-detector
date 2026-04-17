# api/sources/bluesky_client.py
"""
Bluesky адаптер через офіційну atproto Python SDK.

Реєстрація:
1. https://bsky.app — створіть акаунт
2. Settings → Privacy and Security → App Passwords → New
3. Збережіть у .env:
     BSKY_HANDLE=your.handle.bsky.social
     BSKY_APP_PASSWORD=xxxx-xxxx-xxxx-xxxx

Без credentials пошук та recent працюють у публічному режимі (без авторизації),
але з обмеженим функціоналом.

Документація: https://atproto.blue
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Optional

from .base import BaseNewsSource, NewsItem, SourceError

logger = logging.getLogger(__name__)

# URL формат: https://bsky.app/profile/{handle}/post/{rkey}
BSKY_URL_RE = re.compile(
    r"https?://(?:www\.)?bsky\.app/profile/([^/]+)/post/([^/?#]+)",
    re.IGNORECASE,
)


class BlueskySource(BaseNewsSource):
    source_name = "bluesky"

    def __init__(self):
        self._client = None
        self._authenticated = False

    def _get_client(self):
        """Ледача ініціалізація клієнта."""
        if self._client is not None:
            return self._client

        try:
            from atproto import Client
        except ImportError as e:
            raise SourceError(
                self.source_name,
                "atproto не встановлено. Додайте 'atproto' в requirements.txt",
                http_code=500,
            ) from e

        client = Client()

        handle = os.environ.get("BSKY_HANDLE", "").strip()
        app_password = os.environ.get("BSKY_APP_PASSWORD", "").strip()

        if handle and app_password:
            try:
                client.login(handle, app_password)
                self._authenticated = True
                logger.info(f"Bluesky: authenticated as {handle}")
            except Exception as e:
                logger.warning(
                    f"Bluesky auth failed ({e}), falling back to public mode"
                )
                self._authenticated = False
        else:
            logger.info("Bluesky: no credentials, using public endpoints")

        self._client = client
        return client

    def can_handle_url(self, url: str) -> bool:
        return bool(BSKY_URL_RE.search(url))

    async def search(self, query: str, limit: int = 20) -> list[NewsItem]:
        """Пошук постів через app.bsky.feed.searchPosts."""
        def _sync():
            client = self._get_client()
            try:
                resp = client.app.bsky.feed.search_posts(
                    {"q": query, "limit": min(limit, 100)}
                )
                return resp.posts or []
            except Exception as e:
                msg = str(e)
                if "rate" in msg.lower() or "429" in msg:
                    raise SourceError(
                        self.source_name, "Перевищено rate limit", http_code=429
                    ) from e
                raise SourceError(self.source_name, f"Search failed: {e}") from e

        posts = await asyncio.to_thread(_sync)
        return [self._parse_post(p) for p in posts if self._parse_post(p)]

    async def get_recent(self, limit: int = 20) -> list[NewsItem]:
        """
        Публічний feed "what's hot" (популярні пости).
        Потребує авторизації — якщо її немає, повертаємо порожній список
        з повідомленням у лог.
        """
        def _sync():
            client = self._get_client()
            if not self._authenticated:
                logger.info("Bluesky recent: requires auth, returning empty")
                return []
            try:
                # Public "what's hot" feed URI
                # Див. https://docs.bsky.app/docs/tutorials/viewing-feeds
                feed_uri = "at://did:plc:z72i7hdynmk6r22z27h6tvur/app.bsky.feed.generator/whats-hot"
                resp = client.app.bsky.feed.get_feed(
                    {"feed": feed_uri, "limit": min(limit, 100)}
                )
                return [item.post for item in (resp.feed or [])]
            except Exception as e:
                logger.warning(f"Bluesky get_feed failed: {e}, trying timeline")
                try:
                    resp = client.app.bsky.feed.get_timeline(
                        {"limit": min(limit, 100)}
                    )
                    return [item.post for item in (resp.feed or [])]
                except Exception as e2:
                    raise SourceError(
                        self.source_name, f"get_recent failed: {e2}"
                    ) from e2

        posts = await asyncio.to_thread(_sync)
        return [self._parse_post(p) for p in posts if self._parse_post(p)]

    async def fetch_by_url(self, url: str) -> Optional[NewsItem]:
        """Отримати конкретний пост за URL bsky.app."""
        match = BSKY_URL_RE.search(url)
        if not match:
            return None

        handle, rkey = match.group(1), match.group(2)

        def _sync():
            client = self._get_client()
            try:
                # Спочатку resolve handle → DID
                profile = client.app.bsky.actor.get_profile({"actor": handle})
                did = profile.did

                at_uri = f"at://{did}/app.bsky.feed.post/{rkey}"
                resp = client.app.bsky.feed.get_posts({"uris": [at_uri]})
                posts = resp.posts or []
                return posts[0] if posts else None
            except Exception as e:
                raise SourceError(
                    self.source_name, f"fetch_by_url failed: {e}"
                ) from e

        post = await asyncio.to_thread(_sync)
        return self._parse_post(post) if post else None

    # ── Parsing ───────────────────────────────────────────────────────────

    def _parse_post(self, post) -> Optional[NewsItem]:
        """
        Перетворити PostView з atproto на NewsItem.
        Повертає None, якщо пост має непридатний формат.
        """
        if post is None:
            return None

        try:
            record = post.record
            text = getattr(record, "text", "") or ""
            if not text.strip():
                return None

            author = post.author
            author_name = getattr(author, "display_name", None) or None
            author_handle = getattr(author, "handle", None) or None

            # URL для UI: конвертуємо AT URI на bsky.app лінк
            post_uri = post.uri  # at://did:.../app.bsky.feed.post/rkey
            rkey = post_uri.split("/")[-1] if post_uri else ""
            web_url = (
                f"https://bsky.app/profile/{author_handle}/post/{rkey}"
                if author_handle and rkey
                else ""
            )

            created = getattr(record, "created_at", None)
            lang_list = getattr(record, "langs", None) or []
            language = lang_list[0] if lang_list else None

            return NewsItem(
                id=f"bluesky:{post_uri}",
                source=self.source_name,
                url=web_url,
                text=text,
                title=None,  # Bluesky має тільки text
                author=author_name,
                author_handle=f"@{author_handle}" if author_handle else None,
                created_at=created,
                likes_count=getattr(post, "like_count", None),
                reposts_count=getattr(post, "repost_count", None),
                replies_count=getattr(post, "reply_count", None),
                language=language,
            )
        except Exception as e:
            logger.warning(f"Bluesky parse error: {e}, skipping post")
            return None
