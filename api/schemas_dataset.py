"""
Pydantic schemas for Dataset module.
Add these to your existing api/schemas.py (or import from this file).
"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


# ── Dataset CRUD ────────────────────────────────────────────────────────

class DatasetComponents(BaseModel):
    """Which CSV components are present in a dataset."""
    news: bool = True
    tweets: bool = False
    retweets: bool = False
    replies: bool = False
    likes: bool = False
    users: bool = False
    evidence: bool = False


class DatasetResponse(BaseModel):
    """Full dataset info for listing and details."""
    id: int
    user_id: int
    name: str
    description: Optional[str] = None

    total_news: int
    fake_count: int
    real_count: int
    unlabeled_count: int

    has_news: bool
    has_tweets: bool
    has_retweets: bool
    has_replies: bool
    has_likes: bool
    has_users: bool
    has_evidence: bool

    file_size_bytes: int
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class DatasetUploadResponse(BaseModel):
    """Returned after successful upload + validation."""
    dataset: DatasetResponse
    preview: dict  # {"news_head": [...], "counts": {...}}
    warnings: list[str] = []  # non-fatal issues found during validation


class DatasetStatsResponse(BaseModel):
    """Detailed statistics returned on GET /datasets/{id}/stats."""
    dataset_id: int
    total_news: int
    fake_count: int
    real_count: int
    unlabeled_count: int

    # Text statistics (from news.csv)
    text_length_stats: dict  # {"min", "median", "mean", "max", "p95"}
    empty_text_count: int
    duplicate_text_count: int

    # Tweets stats (if has_tweets)
    total_tweets: Optional[int] = None
    avg_tweets_per_news: Optional[float] = None
    news_with_tweets_pct: Optional[float] = None

    # Engagement (if has_likes/retweets/replies)
    total_likes: Optional[int] = None
    total_retweets: Optional[int] = None
    total_replies: Optional[int] = None

    # Users (if has_users)
    total_users: Optional[int] = None
    verified_users_pct: Optional[float] = None
    avg_followers_count: Optional[float] = None

    # Domain distribution (top 10 domains from news.csv)
    top_domains: list[dict] = []  # [{"domain": "...", "count": N}, ...]


class DatasetUpdate(BaseModel):
    """Update name/description only."""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=2000)