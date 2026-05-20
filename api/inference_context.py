"""Context builder для моделей, яким для inference потрібно більше за raw text.

Сценарії:
  • NB-article / DistilBERT / LLM   — context НЕ потрібен (claim_extraction=False)
  • NB-aggregated                   — потрібні social aggregates (counts/mean_followers/...)
  • GIN / SAGE                      — потрібен повний граф (article + посилаючі пости + користувачі)

Pipeline на високому рівні:
  1. (opt) claim extraction — стискаємо текст до перевіряваного твердження
  2. (opt) social search    — Bluesky / Mastodon за claim
  3. (opt) social aggregates — для NB-aggregated
  4. (opt) graph construction — для GNN (Phase 1 = placeholder)

Phase 1 (цей файл): кроки 1-3 повноцінні; крок 4 — заглушка з placeholder dict.
Phase 2 (окремий промт): реальна PyG Data серіалізація для GIN/SAGE.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from api.claim_extractor import extract_claims as sync_extract_claims
from api.sources import get_source
from api.sources._relevance import filter_by_relevance

logger = logging.getLogger(__name__)

_CONTEXT_CACHE: dict = {}
_CACHE_TTL = 300.0


def derive_requirements(*, model_type: str, pipeline_type: str | None) -> dict:
    """Стандартні requirements за типом моделі. Використовується:
    - при створенні нового record у /models POST
    - у backfill-скрипті для legacy записів
    """
    if pipeline_type == "aggregated":
        return {
            "claim_extraction": True,
            "social_search": {
                "enabled": True,
                "sources": ["bluesky", "mastodon"],
                "max_posts": 30,
                "lookback_days": 7,
            },
            "social_aggregates": [
                "tweet_count", "mean_followers", "verified_ratio",
                "mean_retweets", "mean_favorites",
            ],
            "graph_construction": {"enabled": False},
        }
    if model_type in ("gin", "sage", "gnn"):
        return {
            "claim_extraction": True,
            "social_search": {
                "enabled": True,
                "sources": ["bluesky", "mastodon"],
                "max_posts": 50,
                "lookback_days": 14,
            },
            "social_aggregates": [],
            "graph_construction": {
                "enabled": True,
                "node_types": ["article", "post", "user"],
                "edge_types": ["posts", "reposts"],
            },
        }
    return {
        "claim_extraction": False,
        "social_search": {"enabled": False},
        "social_aggregates": [],
        "graph_construction": {"enabled": False},
    }


def needs_context(requirements: dict | None) -> bool:
    """Чи треба запускати build_inference_context взагалі."""
    if not requirements:
        return False
    return bool(
        requirements.get("claim_extraction")
        or (requirements.get("social_search") or {}).get("enabled")
        or (requirements.get("graph_construction") or {}).get("enabled")
    )


async def build_inference_context(
    text: str,
    inference_requirements: dict,
    *,
    options: Optional[dict] = None,
) -> dict:
    """Будує контекст для inference. async — щоб не блокувати FastAPI loop
    на social-search HTTP-запитах.

    Returns:
        {
          "text": str,
          "claim": str | None,
          "related_posts": list[dict] | None,
          "aggregates": dict | None,
          "graph_data": dict | None,
          "metadata": {build_time_ms, sources_used, n_posts_found, warnings},
        }
    """
    options = options or {}
    t0 = time.time()
    warnings: list[str] = []
    context: dict[str, Any] = {
        "text": text,
        "claim": None,
        "related_posts": None,
        "aggregates": None,
        "graph_data": None,
        "metadata": {},
    }

    if inference_requirements.get("claim_extraction"):
        try:
            result = sync_extract_claims(text, use_llm=True)
            claims = result.get("claims") or []
            if claims:
                context["claim"] = str(claims[0].get("claim") or "").strip() or None
        except Exception as e:
            logger.warning(f"claim_extraction failed: {e}")
            warnings.append(f"claim_extraction_failed: {e}")
        if not context["claim"]:
            context["claim"] = text[:200]

    search_cfg = inference_requirements.get("social_search") or {}
    sources_used: list[str] = []
    if search_cfg.get("enabled"):
        query = context["claim"] or text[:100]
        sources_list = list(search_cfg.get("sources") or ["bluesky", "mastodon"])
        max_posts = int(search_cfg.get("max_posts", 30))

        cache_key = (query[:200], tuple(sorted(sources_list)))
        cached = _CONTEXT_CACHE.get(cache_key)
        now = time.time()

        if cached and (now - cached[0]) < _CACHE_TTL:
            posts_relevant = cached[1]
            sources_used = sources_list
            logger.info(f"inference_context cache hit: {len(posts_relevant)} posts")
        else:
            posts, sources_used = await _search_across_sources(
                query=query, sources=sources_list, max_posts=max_posts,
            )
            posts_relevant = filter_by_relevance(query, posts, min_score=0.3)
            _CONTEXT_CACHE[cache_key] = (now, posts_relevant)
            logger.info(
                f"inference_context fetched: {len(posts)} raw → "
                f"{len(posts_relevant)} relevant"
            )

        context["related_posts"] = [
            p.to_dict() if hasattr(p, "to_dict") else dict(p)
            for p in posts_relevant
        ]
        if len(posts_relevant) < 5:
            warnings.append(
                f"few_posts_found: only {len(posts_relevant)} relevant posts — "
                "social/graph features will be noisy."
            )

    agg_features = inference_requirements.get("social_aggregates") or []
    if agg_features and context["related_posts"] is not None:
        context["aggregates"] = _compute_social_aggregates(
            context["related_posts"], requested_features=agg_features,
        )

    graph_cfg = inference_requirements.get("graph_construction") or {}
    if graph_cfg.get("enabled"):
        try:
            propagation = await _build_propagation_inputs(
                article_text=text,
                related_posts=context["related_posts"] or [],
                top_k_for_thread=int(graph_cfg.get("top_k_for_thread", 5)),
                max_replies_per_post=int(graph_cfg.get("max_replies_per_post", 10)),
            )
            context["graph_data"] = propagation
            warnings.extend(
                (propagation.get("metadata") or {}).get("warnings") or []
            )
        except Exception as e:
            logger.warning(f"_build_propagation_inputs failed: {e}")
            warnings.append(f"graph_construction_failed: {e}")
            context["graph_data"] = {
                "article_text": text, "tweets": [], "retweets": [], "replies": [],
                "metadata": {"warnings": [str(e)]},
            }

    context["metadata"] = {
        "build_time_ms": int((time.time() - t0) * 1000),
        "sources_used": sources_used,
        "n_posts_found": len(context["related_posts"] or []),
        "warnings": warnings,
    }
    return context


async def _search_across_sources(
    *,
    query: str,
    sources: list[str],
    max_posts: int,
) -> tuple[list, list[str]]:
    """Паралельний search. Повертає (posts, successful_sources)."""
    per_source = max(max_posts // max(len(sources), 1), 5)

    tasks = []
    names = []
    for sname in sources:
        src = get_source(sname)
        if src is None:
            continue
        names.append(sname)
        tasks.append(src.search(query, limit=per_source))

    if not tasks:
        return [], []

    results = await asyncio.gather(*tasks, return_exceptions=True)
    posts: list = []
    successful: list[str] = []
    for name, r in zip(names, results):
        if isinstance(r, Exception):
            logger.warning(f"source {name} failed: {r}")
            continue
        successful.append(name)
        posts.extend(r or [])
    return posts[:max_posts], successful


def _safe_attr(post, attr: str, default=0):
    """Дістати атрибут і з dataclass-Post, і з dict-Post (після to_dict())."""
    if isinstance(post, dict):
        v = post.get(attr, default)
    else:
        v = getattr(post, attr, default)
    return default if v is None else v


def _compute_social_aggregates(posts: list, *, requested_features: list[str]) -> dict:
    """Агрегати з пов'язаних постів. Альяс-маппінг між FakeNewsNet-style
    feature names і Mastodon/Bluesky-style полями NewsItem:

      tweet_count    → len(posts)
      mean_followers → mean(author_followers_count)
      verified_ratio → mean(author_is_verified)
      mean_retweets  → mean(reposts_count)         # Bluesky reposts / Mastodon boosts
      mean_favorites → mean(likes_count)
    """
    import numpy as np

    if not posts:
        return {feat: 0.0 for feat in requested_features}

    result: dict[str, float] = {}
    for feat in requested_features:
        if feat == "tweet_count":
            result[feat] = float(len(posts))
            continue
        if feat == "mean_followers":
            vals = [_safe_attr(p, "author_followers_count", 0) for p in posts]
        elif feat == "verified_ratio":
            vals = [1.0 if _safe_attr(p, "author_is_verified", False) else 0.0 for p in posts]
        elif feat == "mean_retweets":
            vals = [_safe_attr(p, "reposts_count", 0) for p in posts]
        elif feat == "mean_favorites":
            vals = [_safe_attr(p, "likes_count", 0) for p in posts]
        else:
            logger.warning(f"Unknown aggregate feature: {feat}")
            result[feat] = 0.0
            continue
        result[feat] = float(np.mean(vals)) if vals else 0.0
    return result


async def _build_propagation_inputs(
    *,
    article_text: str,
    related_posts: list,
    top_k_for_thread: int = 5,
    max_replies_per_post: int = 10,
) -> dict:
    """Перетворити `related_posts` у структуру для `build_inference_graph`.

    Структура повторює FakeNewsNet schema:
      article (root) ← tweets ← retweets / replies

    Для топ-K твітів fetch'имо `get_post_details` (Bluesky/Mastodon API),
    щоб дістати replies + reposted_by → справжні reply-вузли + synthetic
    retweet-вузли (текст оригіналу, бо API не дає окремого тексту для
    reposter — це pointer на original).

    Edge cases:
      • API не повертає reposters (без auth у Mastodon) але дає `reposts_total`
        → створюємо `min(reposts_total, 10)` placeholder-нод з reason у metadata.
      • `text` поля пустого reply → skip.
      • source без `get_post_details` → warning, тількі tweet-вузли без потомків.

    Returns:
      {article_text, tweets:[{text,platform,post_id,metadata}],
       retweets:[{text,original_tweet_idx,metadata}],
       replies:[{text,parent_tweet_idx,parent_reply_idx,metadata}],
       metadata:{n_tweets, n_retweets, n_replies, synthetic_retweets,
                 platforms, warnings}}
    """
    warnings: list[str] = []

    def _get(p, key, default=None):
        if isinstance(p, dict):
            v = p.get(key, default)
        else:
            v = getattr(p, key, default)
        return default if v is None else v

    tweets: list[dict] = []
    for p in related_posts[: top_k_for_thread * 3]:
        text_val = _get(p, "text", "")
        if not text_val:
            continue
        tweets.append({
            "text": text_val,
            "platform": _get(p, "source", "unknown"),
            "post_id": _get(p, "id", ""),
            "metadata": {
                "author_handle": _get(p, "author_handle"),
                "author_followers": _get(p, "author_followers_count"),
                "like_count": int(_get(p, "likes_count", 0) or 0),
                "repost_count": int(_get(p, "reposts_count", 0) or 0),
                "reply_count": int(_get(p, "replies_count", 0) or 0),
            },
        })

    if not tweets:
        return {
            "article_text": article_text,
            "tweets": [], "retweets": [], "replies": [],
            "metadata": {
                "n_tweets": 0, "n_retweets": 0, "n_replies": 0,
                "synthetic_retweets": 0, "platforms": [],
                "warnings": ["no_tweets_found"],
            },
        }

    retweets: list[dict] = []
    replies: list[dict] = []
    synthetic_retweet_count = 0

    for tweet_idx, tw in enumerate(tweets[:top_k_for_thread]):
        platform = tw["platform"]
        post_id = tw["post_id"]
        if not post_id:
            continue

        src = get_source(platform)
        if src is None or not hasattr(src, "get_post_details"):
            warnings.append(f"{platform}_no_post_details_method")
            continue

        try:
            details = await src.get_post_details(
                post_id,
                max_replies=max_replies_per_post,
                max_likers=0,
                max_reposters=20,
            )
        except Exception as e:
            logger.warning(f"get_post_details failed {platform}:{post_id}: {e}")
            warnings.append(f"{platform}_fetch_failed")
            continue

        if details is None:
            continue

        for reply in (details.replies or [])[:max_replies_per_post]:
            reply_text = (getattr(reply, "text", None) or "").strip()
            if not reply_text:
                continue
            author = getattr(reply, "author", None)
            replies.append({
                "text": reply_text,
                "parent_tweet_idx": tweet_idx,
                "parent_reply_idx": None,
                "metadata": {
                    "platform": platform,
                    "author_handle": getattr(author, "handle", None) if author else None,
                },
            })

        reposters = details.reposted_by or []
        for reposter in reposters[:20]:
            retweets.append({
                "text": tw["text"],
                "original_tweet_idx": tweet_idx,
                "metadata": {
                    "platform": platform,
                    "reposter_handle": getattr(reposter, "handle", None) if reposter else None,
                    "is_synthetic": True,
                },
            })
            synthetic_retweet_count += 1

        if not reposters and getattr(details, "fetched_limits", None):
            reposts_total = (details.fetched_limits or {}).get("reposts_total")
            if reposts_total and reposts_total > 0:
                n_synth = min(int(reposts_total), 10)
                for _ in range(n_synth):
                    retweets.append({
                        "text": tw["text"],
                        "original_tweet_idx": tweet_idx,
                        "metadata": {
                            "platform": platform,
                            "reposter_handle": None,
                            "is_synthetic": True,
                            "reason": "count_only_no_user_list",
                        },
                    })
                    synthetic_retweet_count += 1
                warnings.append(
                    f"{platform}_synthetic_retweets_from_count_only: "
                    f"{n_synth} placeholder reposter vertices added"
                )

    platforms = sorted({t["platform"] for t in tweets})

    return {
        "article_text": article_text,
        "tweets": tweets,
        "retweets": retweets,
        "replies": replies,
        "metadata": {
            "n_tweets": len(tweets),
            "n_retweets": len(retweets),
            "n_replies": len(replies),
            "synthetic_retweets": synthetic_retweet_count,
            "platforms": platforms,
            "warnings": warnings,
        },
    }


def invalidate_cache() -> None:
    """Скинути cache (тести / адмін-команди)."""
    _CONTEXT_CACHE.clear()
