# api/sources/rss_client.py
"""
RSS адаптер через feedparser.

Повністю безкоштовне джерело без лімітів і авторизації. Використовуємо
кураторський список новинних RSS-стрічок (можна розширити у .env).

На відміну від Bluesky/Mastodon, RSS — це САЙТИ, не соцмережі. Тому
метадані типу likes/reposts недоступні, натомість є title + опис.

Документація: https://feedparser.readthedocs.io
"""
from __future__ import annotations

import asyncio
import hashlib
import html
import logging
import os
import re
from typing import Optional

import aiohttp

from .base import BaseNewsSource, NewsItem, SourceError

logger = logging.getLogger(__name__)

# Кураторський список новинних RSS-стрічок різного політичного спектру.
# Для дипломної роботи з виявлення фейків важливо мати різноманітність джерел:
# від мейнстриму до tabloid, що дає більше варіативності для класифікації.
DEFAULT_RSS_FEEDS = [
    # Mainstream (reference credibility)
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://feeds.reuters.com/reuters/topNews",
    "https://www.theguardian.com/world/rss",
    "https://rss.cnn.com/rss/edition.rss",
    # Tech/policy
    "https://feeds.arstechnica.com/arstechnica/index",
    "https://techcrunch.com/feed/",
    # Tabloid-ish (higher fake rate — хороший test subject)
    "https://www.dailymail.co.uk/news/index.rss",
    "https://nypost.com/feed/",
]

# HTML-теги для зняття з описів
HTML_TAG_RE = re.compile(r"<[^>]+>")


class RSSSource(BaseNewsSource):
    source_name = "rss"

    def __init__(self):
        # Дозволяємо override через .env (comma-separated)
        env_feeds = os.environ.get("RSS_FEEDS", "").strip()
        if env_feeds:
            self._feeds = [url.strip() for url in env_feeds.split(",") if url.strip()]
        else:
            self._feeds = DEFAULT_RSS_FEEDS

        logger.info(f"RSS: configured with {len(self._feeds)} feeds")

    def can_handle_url(self, url: str) -> bool:
        """
        RSS не має специфічного URL формату — це просто HTTP посилання на
        новинні статті. Обробляємо будь-який URL, якщо жоден інший адаптер
        його не взяв (router логіка).

        Тут повертаємо True для URL-ів, що НЕ виглядають як Bluesky/Mastodon.
        Остаточне рішення приймає router.
        """
        return url.startswith(("http://", "https://"))

    async def search(self, query: str, limit: int = 20) -> list[NewsItem]:
        """
        RSS не має власного пошуку. Стягуємо всі feeds і фільтруємо локально
        за входженням query у title/description.
        """
        all_items = await self._fetch_all_feeds(limit_per_feed=20)
        query_lower = query.lower()

        matches = []
        for item in all_items:
            text_to_search = (
                (item.title or "") + " " + (item.text or "")
            ).lower()
            if query_lower in text_to_search:
                matches.append(item)
            if len(matches) >= limit:
                break

        return matches

    async def get_recent(self, limit: int = 20) -> list[NewsItem]:
        """
        Повертає останні пости з усіх feeds, приблизно рівномірно розподілені.
        Для простоти: беремо top-N з кожного feed, потім обрізаємо.
        """
        per_feed = max(3, (limit // len(self._feeds)) + 1)
        all_items = await self._fetch_all_feeds(limit_per_feed=per_feed)
        return all_items[:limit]

    async def fetch_by_url(self, url: str) -> Optional[NewsItem]:
        """
        Для RSS це означає: завантажити сторінку статті, спробувати витягти
        текст через простий HTML парсинг.

        Робимо спрощено: повертаємо NewsItem з title і metadescription.
        Для якісного article extraction варто використовувати readability/trafilatura,
        але це додаткова залежність — залишимо як stub з базовим функціоналом.
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status != 200:
                        raise SourceError(
                            self.source_name,
                            f"HTTP {resp.status} для URL: {url}",
                        )
                    html_content = await resp.text()
        except asyncio.TimeoutError as e:
            raise SourceError(self.source_name, f"Таймаут при завантаженні: {url}") from e
        except Exception as e:
            raise SourceError(self.source_name, f"Fetch failed: {e}") from e

        # Простий парсинг: витягуємо <title>, <meta description>, <article> body
        title_match = re.search(
            r"<title[^>]*>([^<]+)</title>", html_content, re.IGNORECASE
        )
        title = html.unescape(title_match.group(1).strip()) if title_match else None

        desc_match = re.search(
            r'<meta\s+(?:name|property)=["\'](?:description|og:description)["\'][^>]*content=["\']([^"\']+)["\']',
            html_content,
            re.IGNORECASE,
        )
        description = (
            html.unescape(desc_match.group(1).strip()) if desc_match else ""
        )

        # Спроба витягти основний текст з <article>
        article_match = re.search(
            r"<article[^>]*>(.*?)</article>", html_content, re.IGNORECASE | re.DOTALL
        )
        if article_match:
            article_text = self._strip_html(article_match.group(1))
            # Обмежуємо до ~3000 символів — достатньо для класифікації
            if len(article_text) > 3000:
                article_text = article_text[:3000] + "…"
        else:
            article_text = description

        full_text = article_text or description or title or ""
        if not full_text.strip():
            return None

        # Генеруємо стабільний ID з URL
        url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:12]

        return NewsItem(
            id=f"rss:{url_hash}",
            source=self.source_name,
            url=url,
            text=full_text,
            title=title,
            author=None,
            author_handle=None,
            created_at=None,
            likes_count=None,
            reposts_count=None,
            replies_count=None,
            language=None,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    async def _fetch_all_feeds(self, limit_per_feed: int = 20) -> list[NewsItem]:
        """Паралельно завантажити всі feeds і об'єднати результати."""
        tasks = [self._fetch_feed(url, limit_per_feed) for url in self._feeds]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        items = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"RSS feed fetch failed: {r}")
                continue
            items.extend(r)

        # Сортуємо за датою (найсвіжіші зверху), None в кінець
        items.sort(
            key=lambda x: x.created_at or "",
            reverse=True,
        )
        return items

    async def _fetch_feed(self, feed_url: str, limit: int) -> list[NewsItem]:
        """Завантажити і розпарсити один RSS feed."""

        def _sync():
            try:
                import feedparser
            except ImportError as e:
                raise SourceError(
                    self.source_name,
                    "feedparser не встановлено. Додайте у requirements.txt",
                    http_code=500,
                ) from e

            # feedparser сам робить HTTP request, але він sync.
            # Обгортаємо у to_thread щоб не блокувати loop.
            feed = feedparser.parse(feed_url)
            if feed.bozo and not feed.entries:
                logger.warning(f"RSS feed bozo: {feed_url} — {feed.bozo_exception}")
                return []

            items = []
            for entry in feed.entries[:limit]:
                item = self._parse_entry(entry, feed_url)
                if item:
                    items.append(item)
            return items

        return await asyncio.to_thread(_sync)

    def _parse_entry(self, entry, feed_url: str) -> Optional[NewsItem]:
        """Перетворити feedparser entry на NewsItem."""
        try:
            title = entry.get("title", "").strip() or None

            # Текст: пробуємо content → summary → description
            content = ""
            if "content" in entry and entry.content:
                content = entry.content[0].get("value", "")
            elif "summary" in entry:
                content = entry.summary
            elif "description" in entry:
                content = entry.description

            text = self._strip_html(content).strip()

            # Комбінуємо title + text для якісної класифікації
            if title and text and not text.startswith(title):
                full_text = f"{title}\n\n{text}"
            else:
                full_text = text or title or ""

            if not full_text.strip():
                return None

            url = entry.get("link", "")
            author = entry.get("author", None)
            published = entry.get("published", None) or entry.get("updated", None)

            # ID — використовуємо entry.id якщо є, інакше hash URL
            entry_id = entry.get("id") or url
            id_hash = hashlib.md5(entry_id.encode("utf-8")).hexdigest()[:12]

            return NewsItem(
                id=f"rss:{id_hash}",
                source=self.source_name,
                url=url,
                text=full_text,
                title=title,
                author=author,
                author_handle=None,
                created_at=published,
                likes_count=None,
                reposts_count=None,
                replies_count=None,
                language=None,
            )
        except Exception as e:
            logger.warning(f"RSS parse error: {e}")
            return None

    @staticmethod
    def _strip_html(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</p>", "\n\n", text, flags=re.IGNORECASE)
        text = HTML_TAG_RE.sub("", text)
        text = html.unescape(text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        return text.strip()
