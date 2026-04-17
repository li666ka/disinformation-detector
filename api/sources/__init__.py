# api/sources/__init__.py
"""
Модуль джерел даних: Bluesky, Mastodon, RSS.

Усі адаптери мають однаковий інтерфейс BaseNewsSource і повертають
NewsItem у нормалізованому форматі.
"""
from .base import BaseNewsSource, NewsItem, SourceError
from .bluesky_client import BlueskySource
from .mastodon_client import MastodonSource
from .rss_client import RSSSource

__all__ = [
    "BaseNewsSource",
    "NewsItem",
    "SourceError",
    "BlueskySource",
    "MastodonSource",
    "RSSSource",
]


# Фабрика: name → instance (ледача ініціалізація, singleton per source)
_SOURCES: dict[str, BaseNewsSource] = {}


def get_source(name: str) -> BaseNewsSource:
    """
    Отримати instance джерела за іменем.
    Кидає ValueError для невідомих імен.
    """
    name = name.lower().strip()
    if name in _SOURCES:
        return _SOURCES[name]

    if name == "bluesky":
        _SOURCES[name] = BlueskySource()
    elif name == "mastodon":
        _SOURCES[name] = MastodonSource()
    elif name == "rss":
        _SOURCES[name] = RSSSource()
    else:
        raise ValueError(f"Unknown source: {name}")

    return _SOURCES[name]


def get_all_sources() -> list[BaseNewsSource]:
    """Усі доступні джерела (для fetch_by_url — перебираємо всі)."""
    return [get_source(name) for name in ("bluesky", "mastodon", "rss")]
