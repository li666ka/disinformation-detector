"""
Unit tests for llm_social_context — runs without dataset files on disk.

Run: pytest tests/test_social_context.py
Or:  python tests/test_social_context.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import api.llm_social_context as sc
from api.llm_social_context import (
    _compute_overview_stats,
    _compute_tweet_impact_score,
    build_social_context,
    clear_cache,
)


def test_overview_stats_basic():
    tweets = pd.DataFrame([
        {
            "tweet_id": "t1", "user_id": "1", "tweet_text": "this is verified evidence",
            "like_count": 100, "tweet_retweet_count": 50, "tweet_created_at": 1700000000,
        },
        {
            "tweet_id": "t2", "user_id": "2", "tweet_text": "this looks fake to me",
            "like_count": 200, "tweet_retweet_count": 100, "tweet_created_at": 1700003600,
        },
    ])
    users = {
        "1": pd.Series({"user_verified": True, "user_followers_count": 10000}),
        "2": pd.Series({"user_verified": False, "user_followers_count": 500}),
    }
    stats = _compute_overview_stats(tweets, pd.DataFrame(), users)

    assert stats["total_tweets"] == 2
    assert stats["total_retweets"] == 150
    assert stats["verified_count"] == 1
    assert stats["sentiment_breakdown"]["supportive"] > 0
    assert stats["sentiment_breakdown"]["skeptical"] > 0


def test_overview_stats_empty():
    stats = _compute_overview_stats(pd.DataFrame(), pd.DataFrame(), {})
    assert stats["total_tweets"] == 0
    assert stats["verified_count"] == 0
    assert stats["cascade_depth"] == 1


def test_overview_cascade_depth():
    """Reply chain of depth 3: tweet → reply1 → reply2."""
    tweets = pd.DataFrame([
        {"tweet_id": "t1", "user_id": "1", "tweet_text": "x", "like_count": 0},
    ])
    replies = pd.DataFrame([
        {"reply_id": "r1", "parent_tweet_id": "t1", "user_id": "2", "like_count": 0},
        {"reply_id": "r2", "parent_tweet_id": "r1", "user_id": "3", "like_count": 0},
    ])
    stats = _compute_overview_stats(tweets, replies, {})
    assert stats["cascade_depth"] == 3


def test_impact_score_verified_boost():
    tweet = pd.Series({"like_count": 100, "tweet_retweet_count": 50, "reply_count": 0})
    user_normal = pd.Series({"user_verified": False, "user_followers_count": 1000})
    user_verified = pd.Series({"user_verified": True, "user_followers_count": 1000})

    score_normal = _compute_tweet_impact_score(tweet, user_normal)
    score_verified = _compute_tweet_impact_score(tweet, user_verified)
    assert score_verified > score_normal


def test_impact_score_engagement_dominates_low_authority():
    """A tweet with many retweets outranks a tweet with no engagement."""
    high = pd.Series({"like_count": 10000, "tweet_retweet_count": 5000, "reply_count": 100})
    low = pd.Series({"like_count": 0, "tweet_retweet_count": 0, "reply_count": 0})
    user = pd.Series({"user_verified": False, "user_followers_count": 100})
    assert _compute_tweet_impact_score(high, user) > _compute_tweet_impact_score(low, user)


def test_impact_score_no_user():
    tweet = pd.Series({"like_count": 50, "tweet_retweet_count": 10, "reply_count": 0})
    score = _compute_tweet_impact_score(tweet, None)
    assert score >= 0
    assert isinstance(score, float)


def _stub_dataset(tweets_by_article, replies_by_tweet, users_lookup, replies=None):
    def _load(_user_id, _dataset_id):
        return {
            "tweets": pd.concat(list(tweets_by_article.values())) if tweets_by_article else None,
            "tweets_by_article": tweets_by_article,
            "replies_by_tweet": replies_by_tweet,
            "replies": replies,
            "users_lookup": users_lookup,
            "users": None,
        }
    return _load


def test_build_social_context_empty_returns_empty_string():
    original = sc._load_dataset_files
    sc._load_dataset_files = _stub_dataset({}, {}, {})
    clear_cache()
    try:
        result = build_social_context(1, 1, "missing_article")
        assert result == ""
    finally:
        sc._load_dataset_files = original


def test_build_social_context_three_levels():
    """Output must contain OVERVIEW, TOP REACTIONS, MINORITY VOICES headers."""
    tweets = pd.DataFrame([
        {
            "article_id": "a1", "tweet_id": "t1", "user_id": "u1",
            "tweet_text": "Confirmed: verified breakthrough evidence reported.",
            "like_count": 500, "tweet_retweet_count": 200, "tweet_created_at": 1700000000,
        },
        {
            "article_id": "a1", "tweet_id": "t2", "user_id": "u2",
            "tweet_text": "Looks supportive: amazing fact check.",
            "like_count": 300, "tweet_retweet_count": 100, "tweet_created_at": 1700001000,
        },
        {
            "article_id": "a1", "tweet_id": "t3", "user_id": "u3",
            "tweet_text": "This is fake and misleading, debunked claim.",
            "like_count": 5, "tweet_retweet_count": 1, "tweet_created_at": 1700002000,
        },
    ])
    users = {
        "u1": pd.Series({
            "user_screen_name": "alice", "user_verified": True,
            "user_followers_count": 50000, "user_description": "journalist",
        }),
        "u2": pd.Series({
            "user_screen_name": "bob", "user_verified": False,
            "user_followers_count": 1500, "user_description": "",
        }),
        "u3": pd.Series({
            "user_screen_name": "carol", "user_verified": False,
            "user_followers_count": 200, "user_description": "",
        }),
    }

    original = sc._load_dataset_files
    sc._load_dataset_files = _stub_dataset(
        {"a1": tweets}, {}, users, replies=pd.DataFrame()
    )
    clear_cache()
    try:
        ctx = build_social_context(
            1, 1, "a1",
            n_tweets=2,
            include_overview=True,
            include_minority_voices=True,
        )
    finally:
        sc._load_dataset_files = original

    assert "[SOCIAL CONTEXT]" in ctx
    assert "=== OVERVIEW ===" in ctx
    assert "=== TOP REACTIONS" in ctx
    assert "=== MINORITY VOICES" in ctx
    assert "@alice" in ctx
    assert "carol" in ctx or "fake" in ctx.lower()


def test_build_social_context_no_minority_voices_section_skipped():
    tweets = pd.DataFrame([
        {
            "article_id": "a1", "tweet_id": "t1", "user_id": "u1",
            "tweet_text": "neutral content here",
            "like_count": 10, "tweet_retweet_count": 1, "tweet_created_at": 1700000000,
        },
    ])
    users = {"u1": pd.Series({"user_screen_name": "alice", "user_verified": False, "user_followers_count": 100})}

    original = sc._load_dataset_files
    sc._load_dataset_files = _stub_dataset({"a1": tweets}, {}, users, replies=pd.DataFrame())
    clear_cache()
    try:
        ctx = build_social_context(1, 1, "a1", include_overview=False, include_minority_voices=False)
    finally:
        sc._load_dataset_files = original

    assert "=== OVERVIEW ===" not in ctx
    assert "=== MINORITY VOICES" not in ctx
    assert "=== TOP REACTIONS" in ctx


def test_build_social_context_backward_compat_default_args():
    """Old callers using only positional/required args still get valid output."""
    tweets = pd.DataFrame([
        {
            "article_id": "a1", "tweet_id": "t1", "user_id": "u1",
            "tweet_text": "content", "like_count": 10, "tweet_retweet_count": 1,
            "tweet_created_at": 1700000000,
        },
    ])
    users = {"u1": pd.Series({"user_screen_name": "alice", "user_verified": False, "user_followers_count": 100})}

    original = sc._load_dataset_files
    sc._load_dataset_files = _stub_dataset({"a1": tweets}, {}, users, replies=pd.DataFrame())
    clear_cache()
    try:
        ctx = build_social_context(1, 1, "a1")
    finally:
        sc._load_dataset_files = original

    assert ctx.startswith("[SOCIAL CONTEXT]")
    assert "@alice" in ctx


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
