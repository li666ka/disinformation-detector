# api/sources/rss_client.py — оновлений список джерел
"""
RSS адаптер через feedparser.

Розширений список англомовних джерел з різних категорій:
- Mainstream news (high credibility)
- Tabloid (високий рівень сенсаційності — корисно для тесту детекції)
- Political (різні кути — лівий/центр/правий)
- Technology
- Health & Science
- Business & Finance

Категоризація важлива для диплому: можна показати, що модель виявляє фейки
з різною частотою залежно від типу джерела.
"""
from __future__ import annotations

import asyncio
import hashlib
import html
import logging
import os
import re
import time
from typing import Optional

import aiohttp

from .base import BaseNewsSource, NewsItem, SourceError

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# КУРАТОРСЬКИЙ СПИСОК RSS ДЖЕРЕЛ
# ──────────────────────────────────────────────────────────────────────────
# Усі feeds англомовні, перевірені 17 квітня 2026.
# Групи позначені коментарями для зручності.
#
# Якщо хочете використовувати підмножину — скопіюйте потрібні URL у .env
# як: RSS_FEEDS=url1,url2,url3
# ──────────────────────────────────────────────────────────────────────────

DEFAULT_RSS_FEEDS = [
    # ═══════ Mainstream International News (високий рейтинг довіри) ═══════
    "https://feeds.bbci.co.uk/news/world/rss.xml",           # BBC World
    "https://www.theguardian.com/world/rss",                 # The Guardian
    "https://feeds.npr.org/1001/rss.xml",                    # NPR News
    "https://feeds.washingtonpost.com/rss/world",            # Washington Post World
    "https://www.aljazeera.com/xml/rss/all.xml",             # Al Jazeera
    "https://feeds.skynews.com/feeds/rss/world.xml",         # Sky News
    "https://abcnews.go.com/abcnews/internationalheadlines", # ABC News International

    # ═══════ Tabloid & Sensational (часто містять misinformation) ═══════
    # Це КОРИСНО для диплому — модель має детектувати більше фейків тут.
    "https://www.dailymail.co.uk/news/index.rss",            # Daily Mail (UK tabloid)
    "https://nypost.com/feed/",                              # NY Post
    "https://www.thesun.co.uk/news/feed/",                   # The Sun (UK tabloid)
    "https://www.mirror.co.uk/news/rss.xml",                 # Daily Mirror

    # ═══════ Political (різні політичні спектри) ═══════
    "https://www.breitbart.com/feed/",                       # Breitbart (right-wing)
    "https://thehill.com/feed/",                             # The Hill (political)
    "https://feeds.foxnews.com/foxnews/latest",              # Fox News
    "https://www.politico.com/rss/politicopicks.xml",        # Politico
    "https://www.motherjones.com/feed/",                     # Mother Jones (progressive)

    # ═══════ Technology (менше політичних фейків, більше фактів) ═══════
    "https://feeds.arstechnica.com/arstechnica/index",       # Ars Technica
    "https://techcrunch.com/feed/",                          # TechCrunch
    "https://www.theverge.com/rss/index.xml",                # The Verge
    "https://www.wired.com/feed/rss",                        # Wired
    "https://feeds.bbci.co.uk/news/technology/rss.xml",      # BBC Technology

    # ═══════ Health & Science (важливо для detection в pandemic era) ═══════
    "https://www.sciencedaily.com/rss/all.xml",              # Science Daily
    "https://feeds.bbci.co.uk/news/health/rss.xml",          # BBC Health
    "https://www.nature.com/nature.rss",                     # Nature
    "https://feeds.newscientist.com/home",                   # New Scientist

    # ═══════ Business & Finance ═══════
    "https://feeds.bbci.co.uk/news/business/rss.xml",        # BBC Business
    "https://feeds.marketwatch.com/marketwatch/topstories/", # MarketWatch
    "https://www.ft.com/rss/home",                           # Financial Times

    # ═══════ Entertainment & Celebrity (базова подібність до GossipCop) ═══════
    # Це було доменом тренування вашої моделі (FakeNewsNet/GossipCop),
    # тому результати тут мають бути найкращі.
    "https://www.tmz.com/rss.xml",                           # TMZ
    "https://www.hollywoodreporter.com/feed",                # Hollywood Reporter
    "https://variety.com/feed/",                             # Variety
    "https://www.usmagazine.com/feed/",                      # Us Weekly

    # ═══════ Alternative & Blogs (потенційно проблематичні) ═══════
    # Включайте на власний розсуд — деякі мають високий рівень misinformation.
    "https://www.infowars.com/rss.xml",                      # InfoWars (conspiracy-leaning)
]

# HTML-теги для зняття з описів
HTML_TAG_RE = re.compile(r"<[^>]+>")

# Токенізатор для пошукового запиту: літери, цифри, дефіс (мін. 2 символи)
_QUERY_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-']{1,}")


class RSSSource(BaseNewsSource):
    source_name = "rss"

    # Feed cache: shared across all calls
    _cache: list[NewsItem] = []
    _cache_ts: float = 0
    _CACHE_TTL = 300  # 5 minutes

    def __init__(self):
        env_feeds = os.environ.get("RSS_FEEDS", "").strip()
        if env_feeds:
            self._feeds = [url.strip() for url in env_feeds.split(",") if url.strip()]
        else:
            self._feeds = DEFAULT_RSS_FEEDS

        logger.info(f"RSS: configured with {len(self._feeds)} feeds")

    def can_handle_url(self, url: str) -> bool:
        return url.startswith(("http://", "https://"))

    async def _get_cached_items(self, limit_per_feed: int = 10) -> list[NewsItem]:
        """Return cached feed items, refreshing if stale."""
        now = time.time()
        if RSSSource._cache and (now - RSSSource._cache_ts) < RSSSource._CACHE_TTL:
            return RSSSource._cache
        items = await self._fetch_all_feeds(limit_per_feed=limit_per_feed)
        RSSSource._cache = items
        RSSSource._cache_ts = time.time()
        return items

    async def search(self, query: str, limit: int = 20) -> list[NewsItem]:
        """
        Пошук у кешованих RSS-стрічках з рейтингом релевантності.

        Scoring:
          +3 за слово у заголовку
          +1 за слово у тексті
          +5 за точну фразу у заголовку
          +2 за точну фразу у тексті
          +2 бонусу, якщо знайдено >= 50% слів запиту
        """
        all_items = await self._get_cached_items(limit_per_feed=10)
        logger.info(f"RSS search: query={query!r}, items_in_cache={len(all_items)}")

        q_norm = query.lower().strip()
        if not q_norm:
            return []

        words = _QUERY_TOKEN_RE.findall(q_norm)
        if not words:
            return []

        word_patterns = [re.compile(rf"\b{re.escape(w)}\b") for w in words]
        phrase_pattern = (
            re.compile(rf"\b{re.escape(q_norm)}\b")
            if len(words) > 1 else None
        )
        min_match = max(1, (len(words) + 1) // 2)

        scored: list[tuple[float, NewsItem]] = []
        for item in all_items:
            title_low = (item.title or "").lower()
            text_low = (item.text or "").lower()

            score = 0.0
            matched = 0

            for pat in word_patterns:
                if pat.search(title_low):
                    score += 3.0
                    matched += 1
                elif pat.search(text_low):
                    score += 1.0
                    matched += 1

            if phrase_pattern is not None:
                if phrase_pattern.search(title_low):
                    score += 5.0
                elif phrase_pattern.search(text_low):
                    score += 2.0

            if matched >= min_match:
                score += 2.0

            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda s: s[0], reverse=True)

        logger.info(
            f"RSS search: {len(scored)} items matched, "
            f"top score={scored[0][0] if scored else 0}"
        )
        return [item for _, item in scored[:limit]]

    async def get_recent(self, limit: int = 20) -> list[NewsItem]:
        per_feed = max(2, (limit // len(self._feeds)) + 1)
        all_items = await self._get_cached_items(limit_per_feed=per_feed)
        return all_items[:limit]

    async def fetch_by_url(self, url: str) -> Optional[NewsItem]:
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

        article_match = re.search(
            r"<article[^>]*>(.*?)</article>", html_content, re.IGNORECASE | re.DOTALL
        )
        if article_match:
            article_text = self._strip_html(article_match.group(1))
            if len(article_text) > 3000:
                article_text = article_text[:3000] + "…"
        else:
            article_text = description

        full_text = article_text or description or title or ""
        if not full_text.strip():
            return None

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

    # ── Helpers ──────────────────────────────────────────────────────────

    async def _fetch_all_feeds(self, limit_per_feed: int = 20) -> list[NewsItem]:
        tasks = [self._fetch_feed(url, limit_per_feed) for url in self._feeds]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        items = []
        for url, r in zip(self._feeds, results):
            if isinstance(r, Exception):
                logger.warning(f"RSS feed fetch failed for {url}: {r}")
                continue
            items.extend(r)

        items.sort(
            key=lambda x: x.created_at or "",
            reverse=True,
        )
        return items

    async def _fetch_feed(self, feed_url: str, limit: int) -> list[NewsItem]:
        def _sync():
            try:
                import feedparser
            except ImportError as e:
                raise SourceError(
                    self.source_name,
                    "feedparser не встановлено. Додайте у requirements.txt",
                    http_code=500,
                ) from e

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
        try:
            title = entry.get("title", "").strip() or None

            content = ""
            if "content" in entry and entry.content:
                content = entry.content[0].get("value", "")
            elif "summary" in entry:
                content = entry.summary
            elif "description" in entry:
                content = entry.description

            text = self._strip_html(content).strip()

            if title and text and not text.startswith(title):
                full_text = f"{title}\n\n{text}"
            else:
                full_text = text or title or ""

            if not full_text.strip():
                return None

            url = entry.get("link", "")
            author = entry.get("author", None)
            published = entry.get("published", None) or entry.get("updated", None)

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
                language="en",
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
