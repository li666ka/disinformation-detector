"""Build social context strings for augmented LLM prompts.

3-level structure:
  Level 1 — OVERVIEW: aggregate stats (volume, sentiment, sources, cascade)
  Level 2 — TOP REACTIONS: impact-ranked tweets with threading
  Level 3 — MINORITY VOICES: counter-narratives & skepticism
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATASETS_ROOT = Path("uploaded_datasets")

# Cache: load tweets/users/replies once per dataset
_cache: dict[tuple[int, int], dict] = {}

# Cache for computed overview stats (heavy on large datasets)
# key: (user_id, dataset_id, article_id) → stats dict
_overview_cache: dict[tuple[int, int, str], dict] = {}

ROLE_KEYWORDS = (
    "journalist", "researcher", "doctor", "scientist",
    "professor", "expert", "analyst", "reporter",
)

SUPPORTIVE_KEYWORDS = frozenset({
    "confirmed", "verified", "evidence", "peer-reviewed",
    "breakthrough", "amazing", "true", "fact",
})
SKEPTICAL_KEYWORDS = frozenset({
    "fake", "false", "debunked", "misleading", "doubt",
    "question", "skeptical", "unverified", "claim", "hoax",
})


def _load_dataset_files(user_id: int, dataset_id: int) -> dict:
    """Lazy-load tweets/replies/users for dataset, cached per (user, dataset)."""
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
            result["tweets_by_article"] = dict(tuple(tweets_df.groupby("article_id")))
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
            result["replies_by_tweet"] = dict(tuple(replies_df.groupby("parent_tweet_id")))
            result["replies"] = replies_df

            # Build dataset-wide children index ONCE: parent_id → [reply_id, ...].
            # Used for fast cascade BFS in _compute_overview_stats.
            children_by_parent: dict[str, list[str]] = {}
            for parent, child in zip(
                replies_df["parent_tweet_id"].values,
                replies_df["reply_id"].values,
            ):
                children_by_parent.setdefault(str(parent), []).append(str(child))
            result["children_by_parent"] = children_by_parent
        except Exception as e:
            logger.warning(f"Failed to load replies.csv: {e}")

    users_path = ds_path / "users.csv"
    if users_path.exists():
        try:
            users_df = pd.read_csv(users_path, low_memory=False)
            users_df["user_id"] = users_df["user_id"].astype(str)
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


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def _is_verified(user_row) -> bool:
    if user_row is None:
        return False
    raw = user_row.get("user_verified", False) if hasattr(user_row, "get") else False
    if raw is True:
        return True
    return str(raw).strip().lower() in ("true", "1", "yes")


def _safe_num(value, default: float = 0.0) -> float:
    n = pd.to_numeric(value, errors="coerce")
    if pd.isna(n):
        return default
    return float(n)


def _detect_role(description: str) -> Optional[str]:
    desc = (description or "").lower()
    for kw in ROLE_KEYWORDS:
        if kw in desc:
            return kw
    return None


def _format_followers(count: float) -> str:
    f = int(count)
    if f >= 1_000_000:
        return f"{f/1_000_000:.1f}M followers"
    if f >= 1_000:
        return f"{f/1_000:.1f}K followers"
    return f"{f} followers"


def _format_user_meta(user_row, fallback_name: str = "unknown") -> str:
    """Compact user profile: @name (verified, 50K followers)."""
    if user_row is None:
        return f"@{fallback_name} (no profile)"

    screen_name = str(user_row.get("user_screen_name", "") or fallback_name).strip() or fallback_name
    parts = ["verified" if _is_verified(user_row) else "not verified"]

    created_at = pd.to_numeric(user_row.get("user_created_at"), errors="coerce")
    if pd.notna(created_at):
        age_years = (1577836800.0 - float(created_at)) / (365.25 * 86400)
        if age_years > 0:
            if age_years < 1:
                parts.append(f"{int(age_years * 12)}mo old")
            else:
                parts.append(f"{age_years:.0f}y old")

    followers = pd.to_numeric(user_row.get("user_followers_count"), errors="coerce")
    if pd.notna(followers):
        parts.append(_format_followers(float(followers)))

    return f"@{screen_name} ({', '.join(parts)})"


def _format_user_meta_rich(user_row, fallback_name: str = "user") -> str:
    """Bracketed format used in 3-level output: @name [VERIFIED • 50K followers • role]."""
    if user_row is None:
        return f"@{fallback_name}"

    screen_name = str(user_row.get("user_screen_name", "") or fallback_name).strip() or fallback_name
    verified_str = "VERIFIED" if _is_verified(user_row) else "NOT VERIFIED"

    followers = _safe_num(user_row.get("user_followers_count"))
    fol_str = _format_followers(followers)

    role = _detect_role(str(user_row.get("user_description", "") or ""))
    role_str = f" • {role}" if role else ""

    return f"@{screen_name} [{verified_str} • {fol_str}{role_str}]"


# ─────────────────────────────────────────────────────────────────
# LEVEL 1: OVERVIEW STATS
# ─────────────────────────────────────────────────────────────────

def _bfs_cascade(article_tweet_ids: set, children_by_parent: dict) -> tuple[int, int]:
    """
    BFS from article roots through dataset-wide children index.
    Returns (total_replies_in_cascade, max_depth).
    Visits only reachable nodes — O(reachable), not O(full dataset).
    """
    if not article_tweet_ids or not children_by_parent:
        return 0, 1

    queue = deque((tid, 1) for tid in article_tweet_ids)
    visited = set(article_tweet_ids)
    max_depth = 1
    while queue:
        cur, depth = queue.popleft()
        if depth > max_depth:
            max_depth = depth
        for child in children_by_parent.get(cur, ()):
            if child not in visited:
                visited.add(child)
                queue.append((child, depth + 1))

    total_replies = len(visited) - len(article_tweet_ids)
    return total_replies, max_depth


def _compute_overview_stats(
    article_tweets: pd.DataFrame,
    replies_df_or_children: object,
    users_lookup: dict,
) -> dict:
    """
    Aggregate metrics for the OVERVIEW section.

    `replies_df_or_children` accepts either:
      - dict {parent_id: [child_id, ...]} — fast path (preferred, built once per dataset)
      - pd.DataFrame — legacy path, builds children index on demand (slower)
    """
    stats = {
        "total_tweets": int(len(article_tweets)) if article_tweets is not None else 0,
        "total_retweets": 0,
        "total_replies": 0,
        "time_spread_hours": 0.0,
        "verified_count": 0,
        "avg_followers_verified": 0.0,
        "avg_followers_all": 0.0,
        "sentiment_breakdown": {"supportive": 0.0, "skeptical": 0.0, "neutral": 0.0},
        "cascade_depth": 1,
        "peak_engagement_hours": 0.0,
    }

    if article_tweets is None or len(article_tweets) == 0:
        return stats

    if "tweet_retweet_count" in article_tweets.columns:
        stats["total_retweets"] = int(
            pd.to_numeric(article_tweets["tweet_retweet_count"], errors="coerce").fillna(0).sum()
        )

    # Resolve children index (fast path) or fall back to building from DataFrame.
    article_tweet_ids = (
        set(article_tweets["tweet_id"].astype(str))
        if "tweet_id" in article_tweets.columns else set()
    )
    if isinstance(replies_df_or_children, dict):
        children_by_parent = replies_df_or_children
    elif isinstance(replies_df_or_children, pd.DataFrame) and len(replies_df_or_children) > 0:
        children_by_parent = {}
        for parent, child in zip(
            replies_df_or_children.get("parent_tweet_id", pd.Series(dtype=str)).values,
            replies_df_or_children.get("reply_id", pd.Series(dtype=str)).values,
        ):
            children_by_parent.setdefault(str(parent), []).append(str(child))
    else:
        children_by_parent = {}

    total_replies, cascade_depth = _bfs_cascade(article_tweet_ids, children_by_parent)
    stats["total_replies"] = total_replies
    stats["cascade_depth"] = cascade_depth

    # Time spread + peak engagement
    if "tweet_created_at" in article_tweets.columns:
        timestamps = pd.to_datetime(
            article_tweets["tweet_created_at"], unit="s", errors="coerce"
        ).dropna()

        if len(timestamps) >= 2:
            time_range = timestamps.max() - timestamps.min()
            stats["time_spread_hours"] = time_range.total_seconds() / 3600.0

            timestamps_sorted = timestamps.sort_values()
            first_time = timestamps_sorted.iloc[0]
            hours_since_first = (timestamps_sorted - first_time).dt.total_seconds() / 3600.0
            max_hours = float(hours_since_first.max())
            if max_hours > 0:
                bins = np.arange(0, max_hours + 1, 1.0)
                if len(bins) >= 2:
                    counts, edges = np.histogram(hours_since_first.values, bins=bins)
                    if len(counts) > 0:
                        stats["peak_engagement_hours"] = float(edges[int(counts.argmax())])

    # Verified / followers
    user_ids = article_tweets["user_id"].astype(str).unique() if "user_id" in article_tweets.columns else []
    verified_users: list[float] = []
    all_followers: list[float] = []
    for uid in user_ids:
        user_row = users_lookup.get(uid)
        if user_row is None:
            continue
        followers = pd.to_numeric(user_row.get("user_followers_count", 0), errors="coerce")
        if pd.notna(followers):
            f = float(followers)
            all_followers.append(f)
            if _is_verified(user_row):
                verified_users.append(f)

    stats["verified_count"] = len(verified_users)
    stats["avg_followers_verified"] = float(np.mean(verified_users)) if verified_users else 0.0
    stats["avg_followers_all"] = float(np.mean(all_followers)) if all_followers else 0.0

    # Sentiment (keyword-based placeholder; replace with VADER/NRC for prod)
    supportive = skeptical = 0
    if "tweet_text" in article_tweets.columns:
        for text in article_tweets["tweet_text"].astype(str):
            lower = text.lower()
            if any(kw in lower for kw in SUPPORTIVE_KEYWORDS):
                supportive += 1
            elif any(kw in lower for kw in SKEPTICAL_KEYWORDS):
                skeptical += 1

    total = stats["total_tweets"]
    if total > 0:
        sup_frac = supportive / total
        skep_frac = skeptical / total
        stats["sentiment_breakdown"] = {
            "supportive": round(sup_frac, 2),
            "skeptical": round(skep_frac, 2),
            "neutral": round(max(0.0, 1.0 - sup_frac - skep_frac), 2),
        }

    return stats


# ─────────────────────────────────────────────────────────────────
# IMPACT SCORE
# ─────────────────────────────────────────────────────────────────

def _compute_tweet_impact_score(tweet_row, user_row) -> float:
    """
    Impact score combining engagement (60%), source authority (30%), recency placeholder (10%).
    Tolerant to missing fields — defaults all to 0.
    """
    likes = _safe_num(tweet_row.get("like_count", 0)) if hasattr(tweet_row, "get") else 0.0
    retweets = _safe_num(tweet_row.get("tweet_retweet_count", 0)) if hasattr(tweet_row, "get") else 0.0
    replies = _safe_num(tweet_row.get("reply_count", 0)) if hasattr(tweet_row, "get") else 0.0

    engagement_raw = (retweets * 3.0) + (likes * 1.0) + (replies * 0.5)
    engagement_score = math.log1p(engagement_raw) / 10.0  # ~[0, 1]

    if user_row is not None:
        verified_bonus = 0.3 if _is_verified(user_row) else 0.0
        followers = _safe_num(user_row.get("user_followers_count", 0))
        followers_score = math.log1p(followers) / 15.0
        authority_score = verified_bonus + (followers_score * 0.7)
    else:
        authority_score = 0.0

    recency_score = 0.0  # ranking happens via _impact_score sort, not in scoring

    return float(engagement_score * 0.60 + authority_score * 0.30 + recency_score * 0.10)


# ─────────────────────────────────────────────────────────────────
# MAIN: build_social_context (3-level)
# ─────────────────────────────────────────────────────────────────

def build_social_context(
    user_id: int,
    dataset_id: int,
    article_id: str,
    n_tweets: int = 5,
    n_replies_per_tweet: int = 1,
    max_tweet_chars: int = 250,
    max_reply_chars: int = 150,
    include_overview: bool = True,
    include_minority_voices: bool = True,
) -> str:
    """
    Build enhanced 3-level social context.

    Level 1 — OVERVIEW: aggregate stats (volume, sentiment, sources, cascade)
    Level 2 — TOP REACTIONS: tweets ranked by impact_score, with notable replies
    Level 3 — MINORITY VOICES: counter-narratives outside the top set

    Returns "" if no data available.
    """
    data = _load_dataset_files(user_id, dataset_id)

    tweets_by_article = data.get("tweets_by_article") or {}
    replies_by_tweet = data.get("replies_by_tweet") or {}
    users_lookup = data.get("users_lookup") or {}

    aid_str = str(article_id)
    article_tweets = tweets_by_article.get(aid_str)

    if article_tweets is None or len(article_tweets) == 0:
        return ""

    # Use the prebuilt children index for O(reachable) cascade BFS — avoids iterating
    # the full replies DataFrame on every article (was the 5s/article bottleneck).
    children_by_parent = data.get("children_by_parent") or {}

    parts: list[str] = ["[SOCIAL CONTEXT]"]

    # ── LEVEL 1: OVERVIEW ──────────────────────────────────────
    if include_overview:
        cache_key = (user_id, dataset_id, aid_str)
        stats = _overview_cache.get(cache_key)
        if stats is None:
            t0 = time.perf_counter()
            stats = _compute_overview_stats(article_tweets, children_by_parent, users_lookup)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.debug(
                "overview_stats computed in %.1fms (article=%s, tweets=%d)",
                elapsed_ms, aid_str, stats["total_tweets"],
            )
            _overview_cache[cache_key] = stats

        parts.append("=== OVERVIEW ===")
        parts.append(
            f"• {stats['total_tweets']} tweets, "
            f"{stats['total_retweets']} retweets, "
            f"{stats['total_replies']} replies "
            f"({stats['time_spread_hours']:.1f}h spread)"
        )
        sent = stats["sentiment_breakdown"]
        parts.append(
            f"• Sentiment: {int(sent['supportive']*100)}% supportive, "
            f"{int(sent['skeptical']*100)}% skeptical, "
            f"{int(sent['neutral']*100)}% neutral"
        )
        if stats["verified_count"] > 0:
            parts.append(
                f"• Top sources: {stats['verified_count']} verified accounts "
                f"({stats['avg_followers_verified']/1000:.1f}K avg followers), "
                f"{stats['total_tweets'] - stats['verified_count']} regular users"
            )
        if stats["cascade_depth"] > 1:
            parts.append(
                f"• Cascade depth: {stats['cascade_depth']} levels, "
                f"peak engagement: {stats['peak_engagement_hours']:.1f}h after first tweet"
            )
        parts.append("")

    # ── LEVEL 2: TOP REACTIONS (by impact) ─────────────────────
    # Vectorized impact score — apply(axis=1) was a hot-spot at 5s/article.
    article_tweets = article_tweets.copy()
    n_rows = len(article_tweets)

    def _col_to_arr(col: str) -> np.ndarray:
        if col not in article_tweets.columns:
            return np.zeros(n_rows, dtype=float)
        return pd.to_numeric(article_tweets[col], errors="coerce").fillna(0).to_numpy(dtype=float)

    likes_arr = _col_to_arr("like_count")
    rt_arr = _col_to_arr("tweet_retweet_count")
    rep_arr = _col_to_arr("reply_count")

    engagement_raw = (rt_arr * 3.0) + (likes_arr * 1.0) + (rep_arr * 0.5)
    engagement_score = np.log1p(engagement_raw) / 10.0

    if "user_id" in article_tweets.columns:
        user_ids = article_tweets["user_id"].astype(str).to_numpy()
    else:
        user_ids = np.array([""] * n_rows)
    verified_arr = np.zeros(n_rows, dtype=float)
    followers_arr = np.zeros(n_rows, dtype=float)
    for idx, uid in enumerate(user_ids):
        u = users_lookup.get(uid)
        if u is None:
            continue
        if _is_verified(u):
            verified_arr[idx] = 0.3
        followers_arr[idx] = _safe_num(u.get("user_followers_count", 0))
    authority_score = verified_arr + (np.log1p(followers_arr) / 15.0) * 0.7

    article_tweets["_impact_score"] = engagement_score * 0.60 + authority_score * 0.30
    top_tweets = article_tweets.nlargest(n_tweets, "_impact_score")

    parts.append("=== TOP REACTIONS (by impact) ===")

    # Pre-compute first_tweet_time for relative timestamps
    first_tweet_time = None
    if "tweet_created_at" in article_tweets.columns:
        ts_series = pd.to_datetime(
            article_tweets["tweet_created_at"], unit="s", errors="coerce"
        )
        if ts_series.notna().any():
            first_tweet_time = ts_series.min()

    for idx, (_, twt) in enumerate(top_tweets.iterrows(), 1):
        tweet_text = str(twt.get("tweet_text", "") or "").strip().replace("\n", " ")
        if not tweet_text:
            continue
        if len(tweet_text) > max_tweet_chars:
            tweet_text = tweet_text[: max_tweet_chars - 3] + "..."

        user_row = users_lookup.get(str(twt.get("user_id", "")))
        user_meta = _format_user_meta_rich(user_row)

        likes = int(_safe_num(twt.get("like_count", 0)))
        retweets = int(_safe_num(twt.get("tweet_retweet_count", 0)))

        time_str = ""
        if first_tweet_time is not None and pd.notna(first_tweet_time):
            created_at = pd.to_datetime(twt.get("tweet_created_at"), unit="s", errors="coerce")
            if pd.notna(created_at):
                hours = (created_at - first_tweet_time).total_seconds() / 3600.0
                if hours < 1:
                    time_str = f"Posted {int(hours * 60)}min after first | "
                else:
                    time_str = f"Posted {hours:.1f}h after first | "

        engagement_str = f"{likes:,} likes • {retweets:,} retweets"
        tweet_id_str = str(twt.get("tweet_id", ""))
        tweet_replies = replies_by_tweet.get(tweet_id_str)
        if tweet_replies is not None and len(tweet_replies) > 0:
            engagement_str += f" • {len(tweet_replies)} replies"

        parts.append(f"{idx}. {user_meta}")
        parts.append(f"   {time_str}{engagement_str}")
        parts.append(f'   "{tweet_text}"')

        # Notable thread (top reply) — controlled by n_replies_per_tweet
        if n_replies_per_tweet > 0 and tweet_replies is not None and len(tweet_replies) > 0:
            top_reply = tweet_replies.nlargest(1, "like_count").iloc[0]
            reply_text = str(top_reply.get("reply_text", "") or "").strip().replace("\n", " ")
            if reply_text:
                if len(reply_text) > max_reply_chars:
                    reply_text = reply_text[: max_reply_chars - 3] + "..."
                rep_user_row = users_lookup.get(str(top_reply.get("user_id", "")))
                rep_meta = _format_user_meta_rich(rep_user_row)
                rep_likes = int(_safe_num(top_reply.get("like_count", 0)))

                parts.append("")
                parts.append("   Notable thread:")
                parts.append(f"   ↳ {rep_meta}:")
                parts.append(f'     "{reply_text}"')
                parts.append(f"     {rep_likes:,} likes")

        parts.append("")

    # ── LEVEL 3: MINORITY VOICES ───────────────────────────────
    if include_minority_voices:
        top_tweet_ids = set(top_tweets["tweet_id"].astype(str)) if "tweet_id" in top_tweets.columns else set()

        minority_tweets: list = []
        if "tweet_text" in article_tweets.columns:
            for _, tweet in article_tweets.iterrows():
                tid = str(tweet.get("tweet_id", ""))
                if tid in top_tweet_ids:
                    continue
                text = str(tweet.get("tweet_text", "")).lower()
                if any(kw in text for kw in SKEPTICAL_KEYWORDS):
                    minority_tweets.append(tweet)

        parts.append("=== MINORITY VOICES (potential counters) ===")
        if minority_tweets:
            def _user_followers(t) -> float:
                u = users_lookup.get(str(t.get("user_id", "")))
                if u is None:
                    return 0.0
                return _safe_num(u.get("user_followers_count", 0))

            low_followers = sum(1 for t in minority_tweets if _user_followers(t) < 1000)
            verified_minority = sum(
                1 for t in minority_tweets
                if _is_verified(users_lookup.get(str(t.get("user_id", ""))))
            )

            if low_followers > 0:
                parts.append(
                    f"• {low_followers} users questioned claims "
                    f"(avg <1K followers, no verification)"
                )
            if verified_minority > 0:
                parts.append(
                    f"• {verified_minority} verified accounts expressed skepticism"
                )

            minority_sorted = sorted(
                minority_tweets,
                key=lambda t: len(str(t.get("tweet_text", ""))),
                reverse=True,
            )[:2]
            for t in minority_sorted:
                text = str(t.get("tweet_text", "")).strip()[:200]
                user_row = users_lookup.get(str(t.get("user_id", "")))
                name = (user_row.get("user_screen_name", "user") if user_row is not None else "user") or "user"
                parts.append(f'• @{name}: "{text}..."')
        else:
            parts.append("• No significant counter-narratives found in dataset")

    return "\n".join(parts)


def clear_cache() -> None:
    """Clear caches (call after dataset re-upload)."""
    _cache.clear()
    _overview_cache.clear()
