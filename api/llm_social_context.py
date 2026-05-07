"""Build social context strings для augmented LLM prompts."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DATASETS_ROOT = Path("uploaded_datasets")

# Cache: завантажити tweets/users/replies один раз на dataset
_cache: dict[tuple[int, int], dict] = {}


def _load_dataset_files(user_id: int, dataset_id: int) -> dict:
    """Lazy-load tweets/replies/users для dataset з cache."""
    key = (user_id, dataset_id)
    if key in _cache:
        return _cache[key]

    ds_path = DATASETS_ROOT / f"user_{user_id}" / f"dataset_{dataset_id}"

    result: dict = {"tweets": None, "replies": None, "users": None}

    tweets_path = ds_path / "tweets.csv"
    if tweets_path.exists():
        try:
            tweets_df = pd.read_csv(tweets_path, low_memory=False)
            tweets_df["article_id"] = tweets_df["article_id"].astype(str)
            tweets_df["tweet_id"] = tweets_df["tweet_id"].astype(str)
            tweets_df["user_id"] = tweets_df["user_id"].astype(str)
            tweets_df["like_count"] = pd.to_numeric(
                tweets_df["like_count"], errors="coerce"
            ).fillna(0)
            # Pre-group by article_id для O(1) lookup
            result["tweets_by_article"] = dict(tuple(
                tweets_df.groupby("article_id")
            ))
            result["tweets"] = tweets_df
        except Exception as e:
            logger.warning(f"Failed to load tweets.csv: {e}")

    replies_path = ds_path / "replies.csv"
    if replies_path.exists():
        try:
            replies_df = pd.read_csv(replies_path, low_memory=False)
            replies_df["parent_tweet_id"] = replies_df["parent_tweet_id"].astype(str)
            replies_df["reply_id"] = replies_df["reply_id"].astype(str)
            replies_df["user_id"] = replies_df["user_id"].astype(str)
            replies_df["like_count"] = pd.to_numeric(
                replies_df["like_count"], errors="coerce"
            ).fillna(0)
            # Pre-group by parent_tweet_id
            result["replies_by_tweet"] = dict(tuple(
                replies_df.groupby("parent_tweet_id")
            ))
            result["replies"] = replies_df
        except Exception as e:
            logger.warning(f"Failed to load replies.csv: {e}")

    users_path = ds_path / "users.csv"
    if users_path.exists():
        try:
            users_df = pd.read_csv(users_path, low_memory=False)
            users_df["user_id"] = users_df["user_id"].astype(str)
            # Build O(1) lookup dict (use to_dict for speed)
            result["users_lookup"] = {
                row["user_id"]: row
                for _, row in users_df.iterrows()
            }
            result["users"] = users_df
        except Exception as e:
            logger.warning(f"Failed to load users.csv: {e}")

    _cache[key] = result
    logger.info(
        f"Loaded social data for dataset {dataset_id}: "
        f"tweets={result['tweets'] is not None}, "
        f"replies={result['replies'] is not None}, "
        f"users={result['users'] is not None}"
    )
    return result


def _format_user_meta(user_row: Optional[dict], fallback_name: str = "unknown") -> str:
    """Форматувати профіль користувача компактно:
    @username (verified, 5y, 50K followers)"""
    if user_row is None:
        return f"@{fallback_name} (no profile)"

    screen_name = str(user_row.get("user_screen_name", "") or fallback_name).strip()
    if not screen_name:
        screen_name = fallback_name

    parts = []

    # Verified
    verified_raw = user_row.get("user_verified", False)
    is_verified = (
        verified_raw is True or
        str(verified_raw).strip().lower() in ("true", "1", "yes")
    )
    parts.append("verified" if is_verified else "not verified")

    # Age у роках (з created_at Unix timestamp)
    created_at = pd.to_numeric(user_row.get("user_created_at"), errors="coerce")
    if pd.notna(created_at):
        # Reference: 2020-01-01
        age_seconds = 1577836800.0 - float(created_at)
        age_years = age_seconds / (365.25 * 86400)
        if age_years > 0:
            if age_years < 1:
                months = int(age_years * 12)
                parts.append(f"{months}mo old")
            else:
                parts.append(f"{age_years:.0f}y old")

    # Followers (compact format: 1.2K, 200M)
    followers = pd.to_numeric(user_row.get("user_followers_count"), errors="coerce")
    if pd.notna(followers):
        f = int(followers)
        if f >= 1_000_000:
            parts.append(f"{f/1_000_000:.1f}M followers")
        elif f >= 1_000:
            parts.append(f"{f/1_000:.1f}K followers")
        else:
            parts.append(f"{f} followers")

    return f"@{screen_name} ({', '.join(parts)})"


def build_social_context(
    user_id: int,
    dataset_id: int,
    article_id: str,
    n_tweets: int = 5,
    n_replies_per_tweet: int = 1,
    max_tweet_chars: int = 200,
    max_reply_chars: int = 150,
) -> str:
    """Build social context string для augmented prompt.

    Returns empty string якщо дані недоступні.
    """
    data = _load_dataset_files(user_id, dataset_id)

    tweets_by_article = data.get("tweets_by_article", {})
    replies_by_tweet = data.get("replies_by_tweet", {})
    users_lookup = data.get("users_lookup", {})

    aid_str = str(article_id)
    article_tweets = tweets_by_article.get(aid_str)

    if article_tweets is None or len(article_tweets) == 0:
        return ""  # Немає твітів для статті

    # Top N tweets by like_count
    top_tweets = article_tweets.nlargest(n_tweets, "like_count")

    parts = ["[SOCIAL CONTEXT]", "Top reactions on Twitter:", ""]

    for _, twt in top_tweets.iterrows():
        tweet_text = str(twt.get("tweet_text", "") or "").strip().replace("\n", " ")
        if len(tweet_text) > max_tweet_chars:
            tweet_text = tweet_text[:max_tweet_chars - 3] + "..."

        if not tweet_text:
            continue

        user_id_str = str(twt.get("user_id", ""))
        user_row = users_lookup.get(user_id_str)
        fallback_name = str(twt.get("tweet_user_name", "user") or "user").replace(" ", "_")
        user_meta = _format_user_meta(user_row, fallback_name)

        parts.append(f'{user_meta}:')
        parts.append(f'  "{tweet_text}"')

        # Find top replies
        tweet_id_str = str(twt.get("tweet_id", ""))
        tweet_replies = replies_by_tweet.get(tweet_id_str)
        if tweet_replies is not None and len(tweet_replies) > 0:
            top_replies = tweet_replies.nlargest(n_replies_per_tweet, "like_count")
            for _, rep in top_replies.iterrows():
                reply_text = str(rep.get("reply_text", "") or "").strip().replace("\n", " ")
                if not reply_text:
                    continue
                if len(reply_text) > max_reply_chars:
                    reply_text = reply_text[:max_reply_chars - 3] + "..."

                rep_user_id = str(rep.get("user_id", ""))
                rep_user_row = users_lookup.get(rep_user_id)
                rep_user_meta = _format_user_meta(rep_user_row, "user")

                parts.append(f'  ↳ Reply by {rep_user_meta}: "{reply_text}"')

        parts.append("")  # blank line between tweets

    return "\n".join(parts)


def clear_cache() -> None:
    """Очистити кеш (на випадок зміни даних)."""
    _cache.clear()
