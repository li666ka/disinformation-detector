# api/sources/mastodon_client.py
"""
Mastodon адаптер.

Ключова перевага Mastodon: публічні endpoints (timeline, search) працюють
БЕЗ авторизації для читання. Тому адаптер не потребує налаштування credentials.

Використовуємо бібліотеку Mastodon.py — найпопулярніший wrapper.
Якщо у .env є MASTODON_ACCESS_TOKEN, використовуємо його (вищі rate limits).

Документація: https://mastodonpy.readthedocs.io
"""
from __future__ import annotations

import asyncio
import html
import logging
import os
import re
from typing import Optional

from .base import BaseNewsSource, NewsItem, SourceError

logger = logging.getLogger(__name__)

DEFAULT_INSTANCE = "https://mastodon.social"

# URL формат: https://{instance}/@{username}/{status_id}
MASTODON_URL_RE = re.compile(
    r"https?://([a-z0-9.-]+)/@([^/]+)/(\d+)",
    re.IGNORECASE,
)

# Простий cleanup HTML-тегів у контенті постів
HTML_TAG_RE = re.compile(r"<[^>]+>")


class MastodonSource(BaseNewsSource):
    source_name = "mastodon"

    def __init__(self):
        self._client = None
        self._instance = os.environ.get(
            "MASTODON_INSTANCE", DEFAULT_INSTANCE
        ).rstrip("/")

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            from mastodon import Mastodon
        except ImportError as e:
            raise SourceError(
                self.source_name,
                "Mastodon.py не встановлено. Додайте 'Mastodon.py' у requirements.txt",
                http_code=500,
            ) from e

        access_token = os.environ.get("MASTODON_ACCESS_TOKEN", "").strip() or None

        client = Mastodon(
            api_base_url=self._instance,
            access_token=access_token,
            ratelimit_method="throw",  # кидає MastodonRatelimitError замість sleep
            request_timeout=30,
        )

        if access_token:
            logger.info(f"Mastodon: authenticated on {self._instance}")
        else:
            logger.info(f"Mastodon: public mode on {self._instance}")

        self._client = client
        return client

    def can_handle_url(self, url: str) -> bool:
        # Mastodon — децентралізований, URL може бути з будь-якого instance.
        # Перевіряємо лише pattern, не domain.
        match = MASTODON_URL_RE.search(url)
        if not match:
            return False
        # Додаткова евристика: ID статусу зазвичай 15-20+ цифр
        status_id = match.group(3)
        return len(status_id) >= 10

    async def search(self, query: str, limit: int = 20) -> list[NewsItem]:
        """Пошук постів через /api/v2/search (type=statuses)."""
        def _sync():
            client = self._get_client()
            try:
                # search_v2 повертає {'accounts', 'statuses', 'hashtags'}
                result = client.search_v2(
                    q=query,
                    result_type="statuses",
                    limit=min(limit, 40),
                )
                return result.get("statuses", []) if result else []
            except Exception as e:
                msg = str(e).lower()
                if "rate" in msg or "429" in msg:
                    raise SourceError(
                        self.source_name,
                        "Перевищено rate limit Mastodon",
                        http_code=429,
                    ) from e
                # Пошук статусів потребує авторизації на більшості instances
                if "401" in msg or "unauthorized" in msg:
                    raise SourceError(
                        self.source_name,
                        "Пошук статусів потребує MASTODON_ACCESS_TOKEN у .env",
                        http_code=401,
                    ) from e
                raise SourceError(
                    self.source_name, f"Search failed: {e}"
                ) from e

        statuses = await asyncio.to_thread(_sync)
        return [self._parse_status(s) for s in statuses if self._parse_status(s)]

    async def get_recent(self, limit: int = 20) -> list[NewsItem]:
        """Публічна стрічка — працює без авторизації."""
        def _sync():
            client = self._get_client()
            try:
                # timeline_public — локальні пости з instance
                # local=False для federated (всі пости з усіх серверів)
                statuses = client.timeline_public(
                    limit=min(limit, 40),
                    local=False,
                )
                return statuses or []
            except Exception as e:
                raise SourceError(
                    self.source_name, f"get_recent failed: {e}"
                ) from e

        statuses = await asyncio.to_thread(_sync)
        return [self._parse_status(s) for s in statuses if self._parse_status(s)]

    async def fetch_by_url(self, url: str) -> Optional[NewsItem]:
        """
        Отримати пост за URL. Якщо instance з URL != нашому — робимо
        cross-instance lookup через search_v2 (резолвить URL до статусу).
        """
        match = MASTODON_URL_RE.search(url)
        if not match:
            return None

        def _sync():
            client = self._get_client()
            try:
                # search_v2 з q=URL резолвить URL на status (навіть між instances)
                result = client.search_v2(q=url, resolve=True, limit=1)
                statuses = result.get("statuses", []) if result else []
                return statuses[0] if statuses else None
            except Exception as e:
                raise SourceError(
                    self.source_name, f"fetch_by_url failed: {e}"
                ) from e

        status = await asyncio.to_thread(_sync)
        return self._parse_status(status) if status else None

    # ── Parsing ───────────────────────────────────────────────────────────

    def _parse_status(self, status) -> Optional[NewsItem]:
        """
        Mastodon повертає dict з html-контентом у полі 'content'.
        Треба зняти HTML-теги, декодувати entity, зберегти посилання.
        """
        if not status:
            return None

        try:
            content_html = status.get("content", "") or ""
            text = self._strip_html(content_html).strip()

            # Додаємо spoiler_text якщо є (content warning)
            spoiler = status.get("spoiler_text", "") or ""
            if spoiler:
                text = f"[CW: {spoiler}] {text}"

            if not text:
                return None

            account = status.get("account") or {}
            author_display = account.get("display_name") or None
            acct = account.get("acct") or None  # username@instance

            created_at = status.get("created_at")
            if hasattr(created_at, "isoformat"):
                created_at = created_at.isoformat()

            status_id = status.get("id")
            status_url = status.get("url") or ""

            return NewsItem(
                id=f"mastodon:{status_id}",
                source=self.source_name,
                url=status_url,
                text=text,
                title=None,
                author=author_display,
                author_handle=f"@{acct}" if acct else None,
                created_at=str(created_at) if created_at else None,
                likes_count=status.get("favourites_count"),
                reposts_count=status.get("reblogs_count"),
                replies_count=status.get("replies_count"),
                language=status.get("language"),
            )
        except Exception as e:
            logger.warning(f"Mastodon parse error: {e}, skipping status")
            return None

    @staticmethod
    def _strip_html(text: str) -> str:
        """Видалити HTML теги, перетворити <br> на \\n, декодувати entities."""
        if not text:
            return ""
        # <br>, </p> → newline (зберігаємо структуру абзаців)
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</p>", "\n\n", text, flags=re.IGNORECASE)
        # Інші теги геть
        text = HTML_TAG_RE.sub("", text)
        # &amp; → &, &lt; → <, тощо
        text = html.unescape(text)
        # Множинні whitespaces → одинарні
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        return text.strip()
