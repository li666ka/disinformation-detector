# api/sources/base.py
"""
Абстрактний інтерфейс для джерел даних соціальних мереж та новин.

Містить типи для:
- NewsItem — пост/стаття (основний тип)
- UserProfile — повний профіль користувача (автора, лайкаря, репостера)
- Reply — відповідь на пост
- Reposter — користувач який репостнув
- Liker — користувач який лайкнув
- PostDetails — повна інформація про пост з усіма взаємодіями
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────
# User Profile
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class UserProfile:
    """
    Повний профіль користувача (автор поста, лайкер, репостер, автор reply).
    Усі опціональні — різні платформи дають різний набір.
    """
    # Identification
    id: str                                # DID (Bluesky) or acct@instance (Mastodon)
    source: str                            # "bluesky" | "mastodon"
    handle: Optional[str] = None           # @user.bsky.social or @user@instance
    display_name: Optional[str] = None

    # Profile metadata
    description: Optional[str] = None
    avatar_url: Optional[str] = None
    created_at: Optional[str] = None       # ISO-8601 — account creation
    account_age_days: Optional[int] = None # Computed from created_at

    # Social graph
    followers_count: Optional[int] = None
    following_count: Optional[int] = None
    posts_count: Optional[int] = None

    # Trust signals
    is_verified: Optional[bool] = None     # Mastodon: verified_at in fields; Bluesky: N/A
    has_custom_domain: Optional[bool] = None  # Bluesky: handle is custom domain
    is_bot: Optional[bool] = None          # Mastodon has this flag
    has_description: Optional[bool] = None

    # Computed fields (from above)
    followers_following_ratio: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────
# Reply
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class Reply:
    """Відповідь на пост (з текстом і профілем автора відповіді)."""
    id: str                                # Reply post ID
    text: str
    created_at: Optional[str] = None
    author: Optional[UserProfile] = None   # Повний профіль автора відповіді

    # Engagement on the reply itself
    likes_count: Optional[int] = None
    reposts_count: Optional[int] = None
    replies_count: Optional[int] = None   # Nested replies count

    # For UI: URL to view this reply
    url: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.author:
            d["author"] = self.author.to_dict()
        return d


# ──────────────────────────────────────────────────────────────────────────
# PostDetails
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class PostDetails:
    """
    Повна інформація про пост включно з учасниками взаємодій.

    Використовується для deep analysis окремого поста — показує хто саме
    лайкнув, репостнув, відповів.
    """
    # Base post data
    post: "NewsItem"                       # The main post

    # Interactions (profiles)
    replies: list[Reply] = field(default_factory=list)
    reposted_by: list[UserProfile] = field(default_factory=list)
    liked_by: list[UserProfile] = field(default_factory=list)
    quoted_by: list[UserProfile] = field(default_factory=list)  # Bluesky-specific

    # Aggregated analysis of participants
    stats: Optional[dict] = None           # Computed by enricher (see cross_platform.py)

    # Hints about what couldn't be fetched (API limits, privacy)
    # E.g., Bluesky getLikes may return max 100; if total likes > 100, flag this
    fetched_limits: Optional[dict] = None  # {"likes_fetched": 100, "likes_total": 9656, ...}

    def to_dict(self) -> dict:
        return {
            "post": self.post.to_dict(),
            "replies": [r.to_dict() for r in self.replies],
            "reposted_by": [u.to_dict() for u in self.reposted_by],
            "liked_by": [u.to_dict() for u in self.liked_by],
            "quoted_by": [u.to_dict() for u in self.quoted_by],
            "stats": self.stats,
            "fetched_limits": self.fetched_limits,
        }


# ──────────────────────────────────────────────────────────────────────────
# NewsItem (existing, extended)
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class NewsItem:
    """
    Нормалізований формат поста/статті з будь-якого джерела.

    Рівні полів:

    Level 1 (Universal — всі джерела):
      id, source, url, text, title, created_at, language

    Level 2 (Author signals — Bluesky, Mastodon, НЕ RSS):
      author, author_handle, author_account_age_days, author_followers_count,
      author_following_count, author_posts_count, author_is_verified,
      author_has_custom_domain, author_has_description

    Level 3 (Engagement — Bluesky, Mastodon, НЕ RSS):
      likes_count, reposts_count, replies_count, quote_count

    Level 4 (Platform-specific):
      has_url_in_text, has_mentions, is_reply, labels
    """

    # ── Level 1: Universal ─────────────────────────────────────────────
    id: str
    source: str
    url: str
    text: str
    title: Optional[str] = None
    created_at: Optional[str] = None
    language: Optional[str] = None

    # ── Level 2: Author signals ────────────────────────────────────────
    author: Optional[str] = None
    author_handle: Optional[str] = None
    author_account_age_days: Optional[int] = None
    author_followers_count: Optional[int] = None
    author_following_count: Optional[int] = None
    author_posts_count: Optional[int] = None
    author_is_verified: Optional[bool] = None
    author_has_custom_domain: Optional[bool] = None
    author_has_description: Optional[bool] = None

    # ── Level 3: Engagement signals ────────────────────────────────────
    likes_count: Optional[int] = None
    reposts_count: Optional[int] = None
    replies_count: Optional[int] = None
    quote_count: Optional[int] = None

    # ── Level 4: Platform-specific ─────────────────────────────────────
    has_url_in_text: Optional[bool] = None
    has_mentions: Optional[bool] = None
    is_reply: Optional[bool] = None
    labels: Optional[list[str]] = None

    raw_metadata: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────
# Base source class
# ──────────────────────────────────────────────────────────────────────────

class BaseNewsSource(ABC):
    """
    Абстрактний клас для всіх джерел.

    Базові методи (обов'язкові для всіх):
    - search, get_recent, fetch_by_url, can_handle_url

    Розширені методи (опційно — тільки для соц-мереж):
    - get_post_details — повертає PostDetails з replies/likers/reposters
    """

    source_name: str = "base"

    @abstractmethod
    async def search(self, query: str, limit: int = 20) -> list[NewsItem]: ...

    @abstractmethod
    async def get_recent(self, limit: int = 20) -> list[NewsItem]: ...

    @abstractmethod
    async def fetch_by_url(self, url: str) -> Optional[NewsItem]: ...

    @abstractmethod
    def can_handle_url(self, url: str) -> bool: ...

    async def get_post_details(
        self,
        post_id: str,
        *,
        max_replies: int = 50,
        max_likers: int = 100,
        max_reposters: int = 100,
    ) -> Optional[PostDetails]:
        """
        За замовчуванням не підтримується (повертає None).
        Перевизначте у підкласі якщо джерело надає деталі взаємодій.
        """
        return None


class SourceError(Exception):
    """Помилка отримання даних від джерела."""
    def __init__(self, source: str, message: str, *, http_code: int = 502):
        super().__init__(f"[{source}] {message}")
        self.source = source
        self.message = message
        self.http_code = http_code


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

_URL_IN_TEXT_RE = re.compile(r'https?://\S+', re.IGNORECASE)
_MENTION_RE = re.compile(r'@\w+', re.IGNORECASE)


def detect_url_in_text(text: str) -> bool:
    return bool(_URL_IN_TEXT_RE.search(text or ""))


def detect_mentions_in_text(text: str) -> bool:
    return bool(_MENTION_RE.search(text or ""))


def compute_account_age_days(created_at_iso: Optional[str]) -> Optional[int]:
    """ISO datetime → days since creation. None if invalid."""
    if not created_at_iso:
        return None
    try:
        iso = created_at_iso.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = now - dt
        return max(0, delta.days)
    except (ValueError, TypeError):
        return None


def compute_followers_ratio(followers: Optional[int], following: Optional[int]) -> Optional[float]:
    """
    Ratio followers/following. Indicator for bots (typically following >> followers).
    """
    if followers is None or following is None:
        return None
    if following == 0:
        return None if followers == 0 else float("inf")
    return round(followers / following, 2)


def analyze_user_profiles(profiles: list[UserProfile]) -> dict:
    """
    Агрегувати список профілів в статистику для аудиту взаємодій.

    Використовується у PostDetails.stats.
    """
    if not profiles:
        return {
            "total": 0,
        }

    total = len(profiles)

    ages = [p.account_age_days for p in profiles if p.account_age_days is not None]
    followers = [p.followers_count for p in profiles if p.followers_count is not None]

    young_30d = sum(1 for p in profiles if p.account_age_days is not None and p.account_age_days < 30)
    young_90d = sum(1 for p in profiles if p.account_age_days is not None and p.account_age_days < 90)

    verified = sum(1 for p in profiles if p.is_verified is True)
    with_custom_domain = sum(1 for p in profiles if p.has_custom_domain is True)
    with_description = sum(1 for p in profiles if p.has_description is True)
    bots = sum(1 for p in profiles if p.is_bot is True)

    low_followers = sum(1 for p in profiles if p.followers_count is not None and p.followers_count < 50)
    high_ratio_suspicious = sum(
        1 for p in profiles
        if p.followers_count is not None and p.following_count is not None
        and p.following_count > 0 and (p.following_count / max(p.followers_count, 1)) > 10
    )

    return {
        "total": total,
        "verified_count": verified,
        "verified_pct": round(verified / total * 100, 1) if total else 0,
        "custom_domain_count": with_custom_domain,
        "custom_domain_pct": round(with_custom_domain / total * 100, 1) if total else 0,
        "bots_count": bots,
        "with_description_pct": round(with_description / total * 100, 1) if total else 0,
        "avg_account_age_days": round(sum(ages) / len(ages)) if ages else None,
        "median_account_age_days": sorted(ages)[len(ages) // 2] if ages else None,
        "young_accounts_30d_pct": round(young_30d / total * 100, 1) if total else 0,
        "young_accounts_90d_pct": round(young_90d / total * 100, 1) if total else 0,
        "avg_followers_count": round(sum(followers) / len(followers)) if followers else None,
        "median_followers_count": sorted(followers)[len(followers) // 2] if followers else None,
        "low_followers_pct": round(low_followers / total * 100, 1) if total else 0,
        "suspicious_ratio_pct": round(high_ratio_suspicious / total * 100, 1) if total else 0,
    }