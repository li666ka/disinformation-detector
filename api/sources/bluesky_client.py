# api/sources/bluesky_client.py
"""
Bluesky адаптер з повною підтримкою post details:
- get_post_details — replies/likers/reposters/quoters з профілями
- batch profile fetching для ефективності
- кеш профілів на 10 хвилин

API endpoints used:
- app.bsky.feed.searchPosts — пошук
- app.bsky.feed.getTimeline — recent
- app.bsky.feed.getPosts — конкретний пост
- app.bsky.feed.getPostThread — пост з replies
- app.bsky.feed.getLikes — хто лайкнув
- app.bsky.feed.getRepostedBy — хто репостнув
- app.bsky.feed.getQuotes — хто квотнув
- app.bsky.actor.getProfiles — batch профілі (≤25 actors)
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import Optional

from .base import (
    BaseNewsSource,
    NewsItem,
    UserProfile,
    Reply,
    PostDetails,
    SourceError,
    detect_url_in_text,
    detect_mentions_in_text,
    compute_account_age_days,
    compute_followers_ratio,
    analyze_user_profiles,
)

logger = logging.getLogger(__name__)

BSKY_URL_RE = re.compile(
    r"https?://(?:www\.)?bsky\.app/profile/([^/]+)/post/([^/?#]+)",
    re.IGNORECASE,
)

STANDARD_BSKY_TLDS = {"bsky.social", "bsky.app"}

_PROFILE_CACHE: dict[str, tuple[dict, float]] = {}
_PROFILE_CACHE_TTL = 600  # 10 min


class BlueskySource(BaseNewsSource):
    source_name = "bluesky"

    def __init__(self):
        self._client = None
        self._authenticated = False

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            from atproto import Client
        except ImportError as e:
            raise SourceError(
                self.source_name,
                "atproto не встановлено",
                http_code=500,
            ) from e

        client = Client()
        handle = os.environ.get("BSKY_HANDLE", "").strip()
        app_password = os.environ.get("BSKY_APP_PASSWORD", "").strip()

        if handle and app_password:
            try:
                client.login(handle, app_password)
                self._authenticated = True
                logger.info(f"Bluesky: authenticated as {handle}")
            except Exception as e:
                logger.warning(f"Bluesky auth failed ({e}), public mode")
                self._authenticated = False
        else:
            logger.info("Bluesky: no credentials, public mode")

        self._client = client
        return client

    def can_handle_url(self, url: str) -> bool:
        return bool(BSKY_URL_RE.search(url))

    # ── Public: basic methods ───────────────────────────────────────────

    async def search(self, query: str, limit: int = 20) -> list[NewsItem]:
        def _sync():
            client = self._get_client()
            try:
                resp = client.app.bsky.feed.search_posts({"q": query, "limit": min(limit, 100), "lang": "en"})
                return resp.posts or []
            except Exception as e:
                msg = str(e)
                if "429" in msg or "rate" in msg.lower():
                    raise SourceError(self.source_name, "Rate limit", http_code=429) from e
                raise SourceError(self.source_name, f"Search failed: {e}") from e

        posts = await asyncio.to_thread(_sync)
        return await self._parse_posts_enriched(posts)

    async def get_recent(self, limit: int = 20) -> list[NewsItem]:
        def _sync():
            client = self._get_client()
            if not self._authenticated:
                return []
            try:
                feed_uri = "at://did:plc:z72i7hdynmk6r22z27h6tvur/app.bsky.feed.generator/whats-hot"
                resp = client.app.bsky.feed.get_feed({"feed": feed_uri, "limit": min(limit, 100)})
                return [item.post for item in (resp.feed or [])]
            except Exception:
                try:
                    resp = client.app.bsky.feed.get_timeline({"limit": min(limit, 100)})
                    return [item.post for item in (resp.feed or [])]
                except Exception as e:
                    raise SourceError(self.source_name, f"get_recent failed: {e}") from e

        posts = await asyncio.to_thread(_sync)
        return await self._parse_posts_enriched(posts)

    async def fetch_by_url(self, url: str) -> Optional[NewsItem]:
        match = BSKY_URL_RE.search(url)
        if not match:
            return None

        handle, rkey = match.group(1), match.group(2)
        at_uri = await self._resolve_at_uri(handle, rkey)
        if not at_uri:
            return None

        def _sync():
            client = self._get_client()
            try:
                resp = client.app.bsky.feed.get_posts({"uris": [at_uri]})
                return (resp.posts or [None])[0]
            except Exception as e:
                raise SourceError(self.source_name, f"fetch_by_url failed: {e}") from e

        post = await asyncio.to_thread(_sync)
        if not post:
            return None
        items = await self._parse_posts_enriched([post])
        return items[0] if items else None

    # ── Public: post details with full interactions ─────────────────────

    async def get_post_details(
        self,
        post_id: str,
        *,
        max_replies: int = 50,
        max_likers: int = 100,
        max_reposters: int = 100,
    ) -> Optional[PostDetails]:
        """
        post_id format: "bluesky:at://did:.../app.bsky.feed.post/rkey"
        or just "at://..."
        or a bsky.app URL.

        Returns PostDetails with replies, reposted_by, liked_by, quoted_by.
        """
        at_uri = await self._resolve_post_id(post_id)
        if not at_uri:
            return None

        # Fetch all interactions concurrently
        post_task = asyncio.create_task(self._fetch_post_with_thread(at_uri, max_replies))
        likes_task = asyncio.create_task(self._fetch_likes(at_uri, max_likers))
        reposts_task = asyncio.create_task(self._fetch_reposted_by(at_uri, max_reposters))
        quotes_task = asyncio.create_task(self._fetch_quotes(at_uri, max_reposters))

        post, replies, likes_total = await post_task
        if not post:
            return None

        liked_by, likes_fetched = await likes_task
        reposted_by, reposts_fetched = await reposts_task
        quoted_by = await quotes_task

        # Enrich reply authors' profiles in batch
        reply_author_dids = [r.author.id for r in replies if r.author and r.author.id]
        if reply_author_dids:
            profiles_map = await self._fetch_profiles_batch(reply_author_dids)
            for r in replies:
                if r.author and r.author.id in profiles_map:
                    self._enrich_profile(r.author, profiles_map[r.author.id])

        # Analyze participants
        all_participants = liked_by + reposted_by + quoted_by + [r.author for r in replies if r.author]

        return PostDetails(
            post=post,
            replies=replies,
            reposted_by=reposted_by,
            liked_by=liked_by,
            quoted_by=quoted_by,
            stats={
                "likers": analyze_user_profiles(liked_by),
                "reposters": analyze_user_profiles(reposted_by),
                "repliers": analyze_user_profiles([r.author for r in replies if r.author]),
                "quoters": analyze_user_profiles(quoted_by),
                "all_participants": analyze_user_profiles(all_participants),
            },
            fetched_limits={
                "likes_fetched": likes_fetched,
                "likes_total": likes_total.get("likes") if likes_total else None,
                "reposts_fetched": reposts_fetched,
                "reposts_total": likes_total.get("reposts") if likes_total else None,
                "replies_fetched": len(replies),
                "replies_total": likes_total.get("replies") if likes_total else None,
            },
        )

    # ── Private: resolve ID ─────────────────────────────────────────────

    async def _resolve_post_id(self, post_id: str) -> Optional[str]:
        """Convert any valid Bluesky post identifier to at:// URI."""
        # Already an at:// URI
        if post_id.startswith("at://"):
            return post_id
        # Prefixed format: "bluesky:at://..."
        if post_id.startswith("bluesky:at://"):
            return post_id[len("bluesky:"):]
        # Web URL
        match = BSKY_URL_RE.search(post_id)
        if match:
            handle, rkey = match.group(1), match.group(2)
            return await self._resolve_at_uri(handle, rkey)
        return None

    async def _resolve_at_uri(self, handle: str, rkey: str) -> Optional[str]:
        """Resolve handle → DID → at:// URI."""
        def _sync():
            client = self._get_client()
            try:
                profile = client.app.bsky.actor.get_profile({"actor": handle})
                return f"at://{profile.did}/app.bsky.feed.post/{rkey}"
            except Exception as e:
                logger.warning(f"resolve_at_uri failed for {handle}: {e}")
                return None

        return await asyncio.to_thread(_sync)

    # ── Private: post thread fetcher ────────────────────────────────────

    async def _fetch_post_with_thread(
        self, at_uri: str, max_replies: int
    ) -> tuple[Optional[NewsItem], list[Reply], dict]:
        """
        Use getPostThread to get both the main post and replies.
        Returns (NewsItem, list[Reply], counts_dict).
        """
        def _sync():
            client = self._get_client()
            try:
                resp = client.app.bsky.feed.get_post_thread({
                    "uri": at_uri,
                    "depth": 1,  # Direct replies only
                    "parentHeight": 0,
                })
                return resp.thread
            except Exception as e:
                logger.warning(f"get_post_thread failed: {e}")
                return None

        thread = await asyncio.to_thread(_sync)
        if not thread or not hasattr(thread, "post"):
            return None, [], {}

        main_post_view = thread.post
        items = await self._parse_posts_enriched([main_post_view])
        news_item = items[0] if items else None

        # Extract replies
        replies_list = []
        raw_replies = getattr(thread, "replies", None) or []
        for r in raw_replies[:max_replies]:
            reply = self._parse_reply_from_thread(r)
            if reply:
                replies_list.append(reply)

        counts = {
            "likes": getattr(main_post_view, "like_count", None),
            "reposts": getattr(main_post_view, "repost_count", None),
            "replies": getattr(main_post_view, "reply_count", None),
        }

        return news_item, replies_list, counts

    def _parse_reply_from_thread(self, thread_node) -> Optional[Reply]:
        """Parse a reply from thread.replies[i] which is a ThreadViewPost."""
        if not thread_node or not hasattr(thread_node, "post"):
            return None

        post_view = thread_node.post
        record = getattr(post_view, "record", None)
        if not record:
            return None

        text = getattr(record, "text", "") or ""
        if not text.strip():
            return None

        author = getattr(post_view, "author", None)
        if not author:
            return None

        author_handle = getattr(author, "handle", None)
        did = getattr(author, "did", None)
        display_name = getattr(author, "display_name", None)

        has_custom = False
        if author_handle:
            has_custom = not any(author_handle.endswith(tld) for tld in STANDARD_BSKY_TLDS)

        author_profile = UserProfile(
            id=did or "",
            source="bluesky",
            handle=f"@{author_handle}" if author_handle else None,
            display_name=display_name,
            has_custom_domain=has_custom,
            # Other fields enriched later via batch profile fetch
        )

        post_uri = getattr(post_view, "uri", "")
        rkey = post_uri.split("/")[-1] if post_uri else ""
        web_url = (
            f"https://bsky.app/profile/{author_handle}/post/{rkey}"
            if author_handle and rkey else None
        )

        return Reply(
            id=post_uri,
            text=text,
            created_at=getattr(record, "created_at", None),
            author=author_profile,
            likes_count=getattr(post_view, "like_count", None),
            reposts_count=getattr(post_view, "repost_count", None),
            replies_count=getattr(post_view, "reply_count", None),
            url=web_url,
        )

    # ── Private: likes fetcher ──────────────────────────────────────────

    async def _fetch_likes(
        self, at_uri: str, max_likers: int
    ) -> tuple[list[UserProfile], int]:
        """
        Paginate through app.bsky.feed.getLikes.
        Returns (profiles, fetched_count).
        """
        def _sync():
            client = self._get_client()
            all_likes = []
            cursor = None
            try:
                while len(all_likes) < max_likers:
                    params = {"uri": at_uri, "limit": min(100, max_likers - len(all_likes))}
                    if cursor:
                        params["cursor"] = cursor
                    resp = client.app.bsky.feed.get_likes(params)
                    batch = resp.likes or []
                    if not batch:
                        break
                    all_likes.extend(batch)
                    cursor = getattr(resp, "cursor", None)
                    if not cursor:
                        break
                return all_likes
            except Exception as e:
                logger.warning(f"get_likes failed: {e}")
                return all_likes

        likes_raw = await asyncio.to_thread(_sync)
        # Each like has .actor which is ProfileView (lite)
        profiles = []
        for like in likes_raw:
            actor = getattr(like, "actor", None)
            if actor:
                profiles.append(self._profile_view_to_user_profile(actor))
        # Enrich with full profile data
        dids = [p.id for p in profiles if p.id]
        if dids:
            full_profiles = await self._fetch_profiles_batch(dids)
            for p in profiles:
                if p.id in full_profiles:
                    self._enrich_profile(p, full_profiles[p.id])
        return profiles, len(profiles)

    # ── Private: reposts fetcher ────────────────────────────────────────

    async def _fetch_reposted_by(
        self, at_uri: str, max_reposters: int
    ) -> tuple[list[UserProfile], int]:
        """Paginate through app.bsky.feed.getRepostedBy."""
        def _sync():
            client = self._get_client()
            all_users = []
            cursor = None
            try:
                while len(all_users) < max_reposters:
                    params = {"uri": at_uri, "limit": min(100, max_reposters - len(all_users))}
                    if cursor:
                        params["cursor"] = cursor
                    resp = client.app.bsky.feed.get_reposted_by(params)
                    batch = resp.reposted_by or []
                    if not batch:
                        break
                    all_users.extend(batch)
                    cursor = getattr(resp, "cursor", None)
                    if not cursor:
                        break
                return all_users
            except Exception as e:
                logger.warning(f"get_reposted_by failed: {e}")
                return all_users

        raw = await asyncio.to_thread(_sync)
        profiles = [self._profile_view_to_user_profile(u) for u in raw]
        dids = [p.id for p in profiles if p.id]
        if dids:
            full_profiles = await self._fetch_profiles_batch(dids)
            for p in profiles:
                if p.id in full_profiles:
                    self._enrich_profile(p, full_profiles[p.id])
        return profiles, len(profiles)

    # ── Private: quotes fetcher ─────────────────────────────────────────

    async def _fetch_quotes(self, at_uri: str, max_quoters: int) -> list[UserProfile]:
        """Fetch quote-posts, extract author profiles."""
        def _sync():
            client = self._get_client()
            all_quotes = []
            cursor = None
            try:
                while len(all_quotes) < max_quoters:
                    params = {"uri": at_uri, "limit": min(100, max_quoters - len(all_quotes))}
                    if cursor:
                        params["cursor"] = cursor
                    resp = client.app.bsky.feed.get_quotes(params)
                    batch = resp.posts or []
                    if not batch:
                        break
                    all_quotes.extend(batch)
                    cursor = getattr(resp, "cursor", None)
                    if not cursor:
                        break
                return all_quotes
            except Exception as e:
                logger.debug(f"get_quotes failed (may be unsupported): {e}")
                return all_quotes

        raw = await asyncio.to_thread(_sync)
        profiles = []
        for post in raw:
            author = getattr(post, "author", None)
            if author:
                profiles.append(self._profile_view_to_user_profile(author))
        dids = [p.id for p in profiles if p.id]
        if dids:
            full_profiles = await self._fetch_profiles_batch(dids)
            for p in profiles:
                if p.id in full_profiles:
                    self._enrich_profile(p, full_profiles[p.id])
        return profiles

    # ── Private: profile utils ──────────────────────────────────────────

    def _profile_view_to_user_profile(self, actor) -> UserProfile:
        """Convert atproto ProfileView (basic) → UserProfile (partial)."""
        did = getattr(actor, "did", "") or ""
        handle = getattr(actor, "handle", None)
        display_name = getattr(actor, "display_name", None)
        has_custom = False
        if handle:
            has_custom = not any(handle.endswith(tld) for tld in STANDARD_BSKY_TLDS)
        return UserProfile(
            id=did,
            source="bluesky",
            handle=f"@{handle}" if handle else None,
            display_name=display_name,
            has_custom_domain=has_custom,
            # Enriched later with batch getProfiles
        )

    def _enrich_profile(self, profile: UserProfile, detail: dict):
        """Fill in detailed fields from a fetched profile dict."""
        profile.description = detail.get("description")
        profile.avatar_url = detail.get("avatar")
        profile.created_at = detail.get("created_at")
        profile.account_age_days = compute_account_age_days(detail.get("created_at"))
        profile.followers_count = detail.get("followers_count")
        profile.following_count = detail.get("follows_count")
        profile.posts_count = detail.get("posts_count")
        profile.has_description = bool(profile.description and profile.description.strip())
        profile.followers_following_ratio = compute_followers_ratio(
            profile.followers_count, profile.following_count
        )

    async def _fetch_profiles_batch(self, dids: list[str]) -> dict[str, dict]:
        """Fetch profiles in batches of 25 DIDs. Returns {did: profile_dict}."""
        if not dids:
            return {}

        now = time.time()
        unique = list(set(dids))
        result = {}
        to_fetch = []

        for did in unique:
            cached = _PROFILE_CACHE.get(did)
            if cached and (now - cached[1]) < _PROFILE_CACHE_TTL:
                result[did] = cached[0]
            else:
                to_fetch.append(did)

        if not to_fetch:
            return result

        def _sync():
            client = self._get_client()
            profiles = []
            for i in range(0, len(to_fetch), 25):
                chunk = to_fetch[i:i + 25]
                try:
                    resp = client.app.bsky.actor.get_profiles({"actors": chunk})
                    profiles.extend(resp.profiles or [])
                except Exception as e:
                    logger.warning(f"batch profiles failed: {e}")
            return profiles

        profiles = await asyncio.to_thread(_sync)
        for p in profiles:
            pdict = {
                "did": getattr(p, "did", None),
                "handle": getattr(p, "handle", None),
                "display_name": getattr(p, "display_name", None),
                "description": getattr(p, "description", None),
                "avatar": getattr(p, "avatar", None),
                "followers_count": getattr(p, "followers_count", None),
                "follows_count": getattr(p, "follows_count", None),
                "posts_count": getattr(p, "posts_count", None),
                "created_at": getattr(p, "created_at", None),
                "indexed_at": getattr(p, "indexed_at", None),
            }
            did = pdict["did"]
            if did:
                _PROFILE_CACHE[did] = (pdict, now)
                result[did] = pdict
        return result

    # ── Parsing (existing) ──────────────────────────────────────────────

    async def _parse_posts_enriched(self, posts) -> list[NewsItem]:
        if not posts:
            return []

        dids = [p.author.did for p in posts if p and p.author and getattr(p.author, "did", None)]
        profiles_map = await self._fetch_profiles_batch(dids)

        items = []
        for post in posts:
            item = self._parse_post(post, profiles_map)
            if item:
                items.append(item)
        return items

    def _parse_post(self, post, profiles_map: dict[str, dict]) -> Optional[NewsItem]:
        if post is None:
            return None

        try:
            record = post.record
            text = getattr(record, "text", "") or ""
            if not text.strip():
                return None

            author = post.author
            did = getattr(author, "did", "") or ""
            author_name = getattr(author, "display_name", None) or None
            author_handle = getattr(author, "handle", None) or None

            post_uri = post.uri
            rkey = post_uri.split("/")[-1] if post_uri else ""
            web_url = (
                f"https://bsky.app/profile/{author_handle}/post/{rkey}"
                if author_handle and rkey else ""
            )

            created = getattr(record, "created_at", None)
            lang_list = getattr(record, "langs", None) or []
            language = lang_list[0] if lang_list else None

            is_reply = getattr(record, "reply", None) is not None

            labels_list = []
            labels_raw = getattr(post, "labels", None) or []
            for lbl in labels_raw:
                val = getattr(lbl, "val", None)
                if val:
                    labels_list.append(val)

            has_custom_domain = False
            if author_handle:
                has_custom_domain = not any(author_handle.endswith(tld) for tld in STANDARD_BSKY_TLDS)

            profile = profiles_map.get(did, {})
            followers = profile.get("followers_count")
            following = profile.get("follows_count")
            posts_count = profile.get("posts_count")
            author_created = profile.get("created_at")
            description = profile.get("description")

            age_days = compute_account_age_days(author_created)
            has_description = bool(description and description.strip())

            return NewsItem(
                id=f"bluesky:{post_uri}",
                source=self.source_name,
                url=web_url,
                text=text,
                title=None,
                created_at=created,
                language=language,

                author=author_name,
                author_handle=f"@{author_handle}" if author_handle else None,
                author_account_age_days=age_days,
                author_followers_count=followers,
                author_following_count=following,
                author_posts_count=posts_count,
                author_is_verified=None,
                author_has_custom_domain=has_custom_domain,
                author_has_description=has_description,

                likes_count=getattr(post, "like_count", None),
                reposts_count=getattr(post, "repost_count", None),
                replies_count=getattr(post, "reply_count", None),
                quote_count=getattr(post, "quote_count", None),

                has_url_in_text=detect_url_in_text(text),
                has_mentions=detect_mentions_in_text(text),
                is_reply=is_reply,
                labels=labels_list if labels_list else None,
            )
        except Exception as e:
            logger.warning(f"Bluesky parse error: {e}, skipping")
            return None