# api/sources/base.py
"""
Абстрактний інтерфейс для джерел даних соціальних мереж та новин.

Усі адаптери (Bluesky, Mastodon, RSS) реалізують цей інтерфейс — роутер та
frontend залишаються незалежними від конкретного джерела.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class NewsItem:
    """
    Нормалізований формат поста/статті з будь-якого джерела.
    Відповідає TypeScript типу NewsItem у ui/src/types.ts.
    """

    # Унікальний ID: "{source}:{native_id}"
    # Напр. "bluesky:at://did:plc:xyz/app.bsky.feed.post/abc"
    id: str

    # "bluesky" | "mastodon" | "rss"
    source: str

    # URL оригінального поста для переходу "Переглянути"
    url: str

    # Обов'язковий текст
    text: str

    # Опціональні поля (None якщо недоступно в конкретному джерелі)
    title: Optional[str] = None
    author: Optional[str] = None
    author_handle: Optional[str] = None
    created_at: Optional[str] = None  # ISO-8601

    # Метадані взаємодій
    likes_count: Optional[int] = None
    reposts_count: Optional[int] = None
    replies_count: Optional[int] = None

    # Визначена мова, якщо відомо ("en", "uk", "ru", ...)
    language: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class BaseNewsSource(ABC):
    """
    Абстрактний клас для всіх джерел. Кожен адаптер реалізує три методи:
    search(), get_recent(), fetch_by_url().

    Усі методи кидають SourceError при помилках — роутер повинен їх ловити
    і перетворювати на HTTP 502/503.
    """

    source_name: str = "base"  # override in subclass

    @abstractmethod
    async def search(self, query: str, limit: int = 20) -> list[NewsItem]:
        """Пошук постів/статей за ключовими словами."""

    @abstractmethod
    async def get_recent(self, limit: int = 20) -> list[NewsItem]:
        """Останні пости/статті без фільтрації."""

    @abstractmethod
    async def fetch_by_url(self, url: str) -> Optional[NewsItem]:
        """
        Отримати конкретний пост за URL.
        Повертає None, якщо URL не належить цьому джерелу.
        """

    @abstractmethod
    def can_handle_url(self, url: str) -> bool:
        """Чи належить URL цьому джерелу?"""


class SourceError(Exception):
    """Помилка отримання даних від джерела (network, parsing, auth, rate-limit)."""

    def __init__(self, source: str, message: str, *, http_code: int = 502):
        super().__init__(f"[{source}] {message}")
        self.source = source
        self.message = message
        self.http_code = http_code
