"""
Pydantic schemas for Dataset module.
Add these to your existing api/schemas.py (or import from this file).
"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


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
    active_split: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class DatasetUploadResponse(BaseModel):
    """Returned after successful upload + validation."""
    dataset: DatasetResponse
    preview: dict
    warnings: list[str] = []


class DatasetStatsResponse(BaseModel):
    """Detailed statistics returned on GET /datasets/{id}/stats."""
    dataset_id: int
    total_news: int
    fake_count: int
    real_count: int
    unlabeled_count: int

    text_length_stats: dict
    empty_text_count: int
    duplicate_text_count: int

    total_tweets: Optional[int] = None
    avg_tweets_per_news: Optional[float] = None
    news_with_tweets_pct: Optional[float] = None

    total_likes: Optional[int] = None
    total_retweets: Optional[int] = None
    total_replies: Optional[int] = None

    total_users: Optional[int] = None
    verified_users_pct: Optional[float] = None
    avg_followers_count: Optional[float] = None

    top_domains: list[dict] = []

    coverage_pct: Optional[float] = None
    coverage_gap_fake_real: Optional[float] = None
    synthetic_articles_count: Optional[int] = None


class DatasetUpdate(BaseModel):
    """Update name/description only."""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=2000)