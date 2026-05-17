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

# In-memory cache: {(text_head, sources_tuple): (timestamp, posts)}
_CONTEXT_CACHE: dict = {}
_CACHE_TTL = 300.0  # 5 хв


# ── Defaults для inference_requirements ──────────────────────────────────


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
    # text-only (NB article, DistilBERT, LLM)
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


# ── Main entry ───────────────────────────────────────────────────────────


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

    # ── 2a. Claim extraction (sync, але швидкий) ─────────────────────
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
            context["claim"] = text[:200]  # graceful fallback

    # ── 2b. Social search (async, cached 5хв) ────────────────────────
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

    # ── 2c. Social aggregates (для NB-aggregated) ────────────────────
    agg_features = inference_requirements.get("social_aggregates") or []
    if agg_features and context["related_posts"] is not None:
        context["aggregates"] = _compute_social_aggregates(
            context["related_posts"], requested_features=agg_features,
        )

    # ── 2d. Graph construction (Phase 1 = placeholder) ───────────────
    graph_cfg = inference_requirements.get("graph_construction") or {}
    if graph_cfg.get("enabled"):
        context["graph_data"] = {
            "placeholder": True,
            "n_nodes": len(context["related_posts"] or []) + 1,
            "n_edges": 0,
            "note": (
                "Phase 1 placeholder. Real PyG Data construction для GIN/SAGE "
                "буде у наступному промті (потребує node/edge feature mapping "
                "з FakeNewsNet schema на Mastodon/Bluesky)."
            ),
        }
        warnings.append("graph_construction_phase1_placeholder")

    context["metadata"] = {
        "build_time_ms": int((time.time() - t0) * 1000),
        "sources_used": sources_used,
        "n_posts_found": len(context["related_posts"] or []),
        "warnings": warnings,
    }
    return context


# ── Helpers ──────────────────────────────────────────────────────────────


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


def invalidate_cache() -> None:
    """Скинути cache (тести / адмін-команди)."""
    _CONTEXT_CACHE.clear()
