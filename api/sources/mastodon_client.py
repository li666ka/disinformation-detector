# api/sources/mastodon_client.py
"""
Mastodon адаптер з повною підтримкою post details.

API endpoints used:
- search_v2 — search
- timeline_public — recent
- status — single status
- /api/v1/statuses/:id/context — parent + descendants (replies)
- /api/v1/statuses/:id/reblogged_by — who boosted
- /api/v1/statuses/:id/favourited_by — who favourited
"""
from __future__ import annotations

import asyncio
import html
import logging
import os
import re
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

MASTODON_URL_RE = re.compile(
    r"https?://(?:[a-z0-9.-]+)/@([\w.-]+)/(\d+)",
    re.IGNORECASE,
)
HTML_TAG_RE = re.compile(r"<[^>]+>")


class MastodonSource(BaseNewsSource):
    source_name = "mastodon"

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            from mastodon import Mastodon
        except ImportError as e:
            raise SourceError(
                self.source_name,
                "Mastodon.py не встановлено",
                http_code=500,
            ) from e

        instance = os.environ.get("MASTODON_INSTANCE", "https://mastodon.social").strip()
        token = os.environ.get("MASTODON_TOKEN", "").strip() or None

        client = Mastodon(api_base_url=instance, access_token=token)
        logger.info(f"Mastodon: instance={instance}, auth={bool(token)}")

        self._client = client
        return client

    def can_handle_url(self, url: str) -> bool:
        return bool(MASTODON_URL_RE.search(url))

    # ── Public: basic ───────────────────────────────────────────────────

    async def search(self, query: str, limit: int = 20) -> list[NewsItem]:
        def _sync():
            client = self._get_client()
            try:
                result = client.search_v2(q=query, result_type="statuses")
                return result.get("statuses", []) if result else []
            except Exception as e:
                msg = str(e)
                if "429" in msg or "rate" in msg.lower():
                    raise SourceError(self.source_name, "Rate limit", http_code=429) from e
                raise SourceError(self.source_name, f"Search failed: {e}") from e

        statuses = await asyncio.to_thread(_sync)
        items = [self._parse_status(s) for s in statuses if self._parse_status(s)]
        return [i for i in items if not i.language or i.language.startswith("en")]

    async def get_recent(self, limit: int = 20) -> list[NewsItem]:
        def _sync():
            client = self._get_client()
            try:
                # mastodon.social disables the public timeline API (returns []),
                # so try it first and fall back to trending statuses
                statuses = client.timeline_public(local=False, limit=limit)
                if statuses:
                    return statuses
                logger.info("Mastodon timeline_public returned empty, falling back to trending")
                statuses = client.trending_statuses(limit=limit)
                return statuses or []
            except Exception as e:
                raise SourceError(self.source_name, f"get_recent failed: {e}") from e

        statuses = await asyncio.to_thread(_sync)
        return [self._parse_status(s) for s in statuses if self._parse_status(s)]

    async def fetch_by_url(self, url: str) -> Optional[NewsItem]:
        match = MASTODON_URL_RE.search(url)
        if not match:
            return None

        def _sync():
            client = self._get_client()
            try:
                result = client.search_v2(q=url, resolve=True, limit=1)
                statuses = result.get("statuses", []) if result else []
                return statuses[0] if statuses else None
            except Exception as e:
                raise SourceError(self.source_name, f"fetch_by_url failed: {e}") from e

        status = await asyncio.to_thread(_sync)
        return self._parse_status(status) if status else None

    # ── Public: post details ────────────────────────────────────────────

    async def get_post_details(
        self,
        post_id: str,
        *,
        max_replies: int = 50,
        max_likers: int = 100,
        max_reposters: int = 100,
    ) -> Optional[PostDetails]:
        """
        post_id format: "mastodon:<id>" or just "<id>" or Mastodon URL.
        Returns PostDetails with context.descendants as replies,
        reblogged_by as reposted_by, favourited_by as liked_by.
        """
        status_id = await self._resolve_status_id(post_id)
        if not status_id:
            return None

        # Fetch status + context + reblogged_by + favourited_by in parallel
        status_task = asyncio.create_task(self._fetch_status(status_id))
        context_task = asyncio.create_task(self._fetch_context(status_id, max_replies))
        reblogged_task = asyncio.create_task(self._fetch_reblogged_by(status_id, max_reposters))
        favourited_task = asyncio.create_task(self._fetch_favourited_by(status_id, max_likers))

        status = await status_task
        if not status:
            return None

        news_item = self._parse_status(status)
        if not news_item:
            return None

        replies, replies_fetched = await context_task
        reposted_by, reposts_fetched = await reblogged_task
        liked_by, likes_fetched = await favourited_task

        all_participants = liked_by + reposted_by + [r.author for r in replies if r.author]

        return PostDetails(
            post=news_item,
            replies=replies,
            reposted_by=reposted_by,
            liked_by=liked_by,
            quoted_by=[],  # Mastodon doesn't have quote posts
            stats={
                "likers": analyze_user_profiles(liked_by),
                "reposters": analyze_user_profiles(reposted_by),
                "repliers": analyze_user_profiles([r.author for r in replies if r.author]),
                "all_participants": analyze_user_profiles(all_participants),
            },
            fetched_limits={
                "likes_fetched": likes_fetched,
                "likes_total": status.get("favourites_count"),
                "reposts_fetched": reposts_fetched,
                "reposts_total": status.get("reblogs_count"),
                "replies_fetched": replies_fetched,
                "replies_total": status.get("replies_count"),
            },
        )

    async def _resolve_status_id(self, post_id: str) -> Optional[str]:
        """Resolve any valid Mastodon identifier to status ID string."""
        if post_id.startswith("mastodon:"):
            return post_id[len("mastodon:"):]

        if re.match(r"^\d+$", post_id):
            return post_id

        # URL form
        match = MASTODON_URL_RE.search(post_id)
        if match:
            # May be on a foreign instance — resolve via search
            def _sync():
                client = self._get_client()
                try:
                    result = client.search_v2(q=post_id, resolve=True, limit=1)
                    statuses = result.get("statuses", []) if result else []
                    if statuses:
                        return str(statuses[0].get("id"))
                except Exception as e:
                    logger.warning(f"mastodon resolve url failed: {e}")
                return None

            return await asyncio.to_thread(_sync)

        return None

    async def _fetch_status(self, status_id: str) -> Optional[dict]:
        def _sync():
            client = self._get_client()
            try:
                return client.status(status_id)
            except Exception as e:
                logger.warning(f"mastodon status fetch failed: {e}")
                return None
        return await asyncio.to_thread(_sync)

    async def _fetch_context(
        self, status_id: str, max_replies: int
    ) -> tuple[list[Reply], int]:
        """Use /api/v1/statuses/:id/context → get descendants (replies)."""
        def _sync():
            client = self._get_client()
            try:
                ctx = client.status_context(status_id)
                return ctx.get("descendants", []) if ctx else []
            except Exception as e:
                logger.warning(f"mastodon context failed: {e}")
                return []

        descendants = await asyncio.to_thread(_sync)
        # Direct replies only (where in_reply_to_id == status_id)
        direct_replies = [
            d for d in descendants
            if str(d.get("in_reply_to_id", "")) == str(status_id)
        ]

        replies = []
        for r in direct_replies[:max_replies]:
            reply = self._parse_reply(r)
            if reply:
                replies.append(reply)

        return replies, len(replies)

    async def _fetch_reblogged_by(
        self, status_id: str, max_reposters: int
    ) -> tuple[list[UserProfile], int]:
        """Use /api/v1/statuses/:id/reblogged_by (paginated)."""
        def _sync():
            client = self._get_client()
            all_accounts = []
            try:
                # Mastodon.py paginates via max_id
                page = client.status_reblogged_by(status_id, limit=min(40, max_reposters))
                while page and len(all_accounts) < max_reposters:
                    all_accounts.extend(page)
                    if len(all_accounts) >= max_reposters:
                        break
                    try:
                        page = client.fetch_next(page)
                    except Exception:
                        break
                return all_accounts
            except Exception as e:
                logger.warning(f"mastodon reblogged_by failed: {e}")
                return all_accounts

        raw = await asyncio.to_thread(_sync)
        profiles = [self._account_to_user_profile(a) for a in raw[:max_reposters]]
        return profiles, len(profiles)

    async def _fetch_favourited_by(
        self, status_id: str, max_likers: int
    ) -> tuple[list[UserProfile], int]:
        """Use /api/v1/statuses/:id/favourited_by (paginated)."""
        def _sync():
            client = self._get_client()
            all_accounts = []
            try:
                page = client.status_favourited_by(status_id, limit=min(40, max_likers))
                while page and len(all_accounts) < max_likers:
                    all_accounts.extend(page)
                    if len(all_accounts) >= max_likers:
                        break
                    try:
                        page = client.fetch_next(page)
                    except Exception:
                        break
                return all_accounts
            except Exception as e:
                logger.warning(f"mastodon favourited_by failed: {e}")
                return all_accounts

        raw = await asyncio.to_thread(_sync)
        profiles = [self._account_to_user_profile(a) for a in raw[:max_likers]]
        return profiles, len(profiles)

    # ── Parsing ──────────────────────────────────────────────────────────

    def _parse_reply(self, status: dict) -> Optional[Reply]:
        """Parse a reply status → Reply."""
        if not status:
            return None

        content_html = status.get("content", "") or ""
        text = self._strip_html(content_html).strip()
        spoiler = status.get("spoiler_text", "") or ""
        if spoiler:
            text = f"[CW: {spoiler}] {text}"

        if not text:
            return None

        account = status.get("account") or {}
        author_profile = self._account_to_user_profile(account)

        created_at = status.get("created_at")
        if hasattr(created_at, "isoformat"):
            created_at = created_at.isoformat()

        return Reply(
            id=str(status.get("id", "")),
            text=text,
            created_at=str(created_at) if created_at else None,
            author=author_profile,
            likes_count=status.get("favourites_count"),
            reposts_count=status.get("reblogs_count"),
            replies_count=status.get("replies_count"),
            url=status.get("url"),
        )

    def _account_to_user_profile(self, account: dict) -> UserProfile:
        """Mastodon account dict → UserProfile."""
        if not account:
            return UserProfile(id="", source="mastodon")

        acct = account.get("acct") or ""
        account_id = str(account.get("id", ""))

        created_at = account.get("created_at")
        if hasattr(created_at, "isoformat"):
            created_at = created_at.isoformat()
        age_days = compute_account_age_days(created_at) if created_at else None

        # Verification via fields[].verified_at
        is_verified = False
        for fld in account.get("fields", []) or []:
            if fld.get("verified_at"):
                is_verified = True
                break

        note = (account.get("note") or "").strip()
        has_desc = bool(self._strip_html(note).strip())

        followers = account.get("followers_count")
        following = account.get("following_count")

        return UserProfile(
            id=account_id,
            source="mastodon",
            handle=f"@{acct}" if acct else None,
            display_name=account.get("display_name"),
            description=self._strip_html(note) if note else None,
            avatar_url=account.get("avatar"),
            created_at=str(created_at) if created_at else None,
            account_age_days=age_days,
            followers_count=followers,
            following_count=following,
            posts_count=account.get("statuses_count"),
            is_verified=is_verified,
            has_custom_domain=None,
            is_bot=account.get("bot", False) or None,
            has_description=has_desc,
            followers_following_ratio=compute_followers_ratio(followers, following),
        )

    def _parse_status(self, status) -> Optional[NewsItem]:
        if not status:
            return None

        try:
            # If this is a boost/reblog, use the original status for content
            reblog = status.get("reblog")
            content_status = reblog if reblog else status

            content_html = content_status.get("content", "") or ""
            text = self._strip_html(content_html).strip()
            spoiler = content_status.get("spoiler_text", "") or ""
            if spoiler:
                text = f"[CW: {spoiler}] {text}"
            if not text:
                return None

            # For reblogs: author is the original poster, not the booster
            account = content_status.get("account") or {}
            author_display = account.get("display_name") or None
            acct = account.get("acct") or None

            created_at = content_status.get("created_at")
            if hasattr(created_at, "isoformat"):
                created_at = created_at.isoformat()

            status_id = content_status.get("id")
            status_url = content_status.get("url") or ""

            author_created = account.get("created_at")
            if hasattr(author_created, "isoformat"):
                author_created = author_created.isoformat()

            age_days = compute_account_age_days(author_created)
            followers = account.get("followers_count")
            following = account.get("following_count")
            posts_count = account.get("statuses_count")
            is_bot = account.get("bot", False)
            note = (account.get("note") or "").strip()
            has_description = bool(self._strip_html(note).strip())

            is_verified = False
            for fld in account.get("fields", []) or []:
                if fld.get("verified_at"):
                    is_verified = True
                    break

            is_reply = bool(content_status.get("in_reply_to_id"))
            mentions_raw = content_status.get("mentions", []) or []
            has_mentions = len(mentions_raw) > 0

            return NewsItem(
                id=f"mastodon:{status_id}",
                source=self.source_name,
                url=status_url,
                text=text,
                title=None,
                created_at=str(created_at) if created_at else None,
                language=content_status.get("language"),

                author=author_display,
                author_handle=f"@{acct}" if acct else None,
                author_account_age_days=age_days,
                author_followers_count=followers,
                author_following_count=following,
                author_posts_count=posts_count,
                author_is_verified=is_verified,
                author_has_custom_domain=None,
                author_has_description=has_description,

                likes_count=content_status.get("favourites_count"),
                reposts_count=content_status.get("reblogs_count"),
                replies_count=content_status.get("replies_count"),
                quote_count=None,

                has_url_in_text=detect_url_in_text(text),
                has_mentions=has_mentions,
                is_reply=is_reply,
                labels=None,
                raw_metadata={"is_bot": is_bot} if is_bot else None,
            )
        except Exception as e:
            logger.warning(f"Mastodon parse error: {e}, skipping")
            return None

    @staticmethod
    def _strip_html(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</p>", "\n\n", text, flags=re.IGNORECASE)
        text = HTML_TAG_RE.sub("", text)
        text = html.unescape(text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        return text.strip()