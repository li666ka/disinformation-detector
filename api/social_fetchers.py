"""Fetch окремих постів з Mastodon та Bluesky за публічним URL.

URL формати:
  - Mastodon: https://<instance>/@<user>/<status_id>
              https://<instance>/<user>/<status_id>
  - Bluesky:  https://bsky.app/profile/<handle>/post/<post_id>
"""
from __future__ import annotations

import html as html_lib
import logging
import re
from dataclasses import dataclass
from typing import Literal, Optional
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)

Platform = Literal["mastodon", "bluesky"]

USER_AGENT = "FakeNewsDetector/1.0 (Academic Research)"


@dataclass
class SocialPost:
    platform: Platform
    url: str
    post_id: str
    author_handle: str
    author_display_name: Optional[str]
    content: str
    created_at: str
    language: Optional[str]

    replies_count: int = 0
    reposts_count: int = 0
    likes_count: int = 0

    media_attachments: Optional[list[dict]] = None
    raw: Optional[dict] = None


def detect_platform(url: str) -> Optional[Platform]:
    if "bsky.app" in url or "bsky.social" in url:
        return "bluesky"
    if re.search(r"/@[\w.-]+/\d+", url):
        return "mastodon"
    return None


def fetch_post(url: str) -> SocialPost:
    platform = detect_platform(url)
    if platform == "mastodon":
        return _fetch_mastodon(url)
    if platform == "bluesky":
        return _fetch_bluesky(url)
    raise ValueError(
        f"Unsupported URL format: {url}. "
        "Supported: Mastodon (https://instance/@user/id), "
        "Bluesky (https://bsky.app/profile/handle/post/id)"
    )


# ── Mastodon ────────────────────────────────────────────────────────

def _fetch_mastodon(url: str) -> SocialPost:
    parsed = urlparse(url)
    instance = parsed.netloc
    match = re.search(r"/(\d+)(?:/|$)", parsed.path)
    if not match:
        raise ValueError(f"Cannot extract status ID from: {url}")
    status_id = match.group(1)

    api_url = f"https://{instance}/api/v1/statuses/{status_id}"
    log.info(f"Fetching Mastodon: {api_url}")
    response = requests.get(
        api_url, timeout=15, headers={"User-Agent": USER_AGENT}
    )
    response.raise_for_status()
    data = response.json()

    account = data.get("account", {}) or {}
    content_plain = _strip_html(data.get("content", ""))

    media = [
        {
            "type": m.get("type"),
            "url": m.get("url"),
            "description": m.get("description"),
        }
        for m in (data.get("media_attachments") or [])
    ]

    return SocialPost(
        platform="mastodon",
        url=data.get("url", url),
        post_id=str(data["id"]),
        author_handle=f"@{account.get('acct', 'unknown')}",
        author_display_name=account.get("display_name"),
        content=content_plain,
        created_at=data.get("created_at", ""),
        language=data.get("language"),
        replies_count=int(data.get("replies_count") or 0),
        reposts_count=int(data.get("reblogs_count") or 0),
        likes_count=int(data.get("favourites_count") or 0),
        media_attachments=media or None,
        raw=data,
    )


def _strip_html(html: str) -> str:
    if not html:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", html)
    text = re.sub(r"</p>", "\n\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html_lib.unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Bluesky ─────────────────────────────────────────────────────────

def _fetch_bluesky(url: str) -> SocialPost:
    match = re.search(r"/profile/([^/]+)/post/([^/?#]+)", url)
    if not match:
        raise ValueError(f"Cannot parse Bluesky URL: {url}")
    handle = match.group(1)
    post_id = match.group(2)

    # Якщо handle — це вже DID (did:plc:...), пропускаємо resolve.
    if handle.startswith("did:"):
        did = handle
    else:
        resolve_url = (
            "https://public.api.bsky.app/xrpc/com.atproto.identity.resolveHandle"
            f"?handle={handle}"
        )
        log.info(f"Resolving Bluesky handle: {handle}")
        response = requests.get(
            resolve_url, timeout=10, headers={"User-Agent": USER_AGENT}
        )
        response.raise_for_status()
        did = response.json()["did"]

    at_uri = f"at://{did}/app.bsky.feed.post/{post_id}"
    thread_url = (
        "https://public.api.bsky.app/xrpc/app.bsky.feed.getPostThread"
        f"?uri={at_uri}&depth=0"
    )
    log.info(f"Fetching Bluesky thread: {thread_url}")
    response = requests.get(
        thread_url, timeout=15, headers={"User-Agent": USER_AGENT}
    )
    response.raise_for_status()
    data = response.json()

    post_data = (data.get("thread") or {}).get("post", {}) or {}
    author = post_data.get("author", {}) or {}
    record = post_data.get("record", {}) or {}

    langs = record.get("langs") or []
    language = langs[0] if langs else None

    return SocialPost(
        platform="bluesky",
        url=url,
        post_id=post_id,
        author_handle=f"@{author.get('handle', 'unknown')}",
        author_display_name=author.get("displayName"),
        content=record.get("text", ""),
        created_at=record.get("createdAt", ""),
        language=language,
        replies_count=int(post_data.get("replyCount") or 0),
        reposts_count=int(post_data.get("repostCount") or 0),
        likes_count=int(post_data.get("likeCount") or 0),
        media_attachments=_extract_bluesky_media(post_data),
        raw=post_data,
    )


def _extract_bluesky_media(post_data: dict) -> Optional[list[dict]]:
    embed = post_data.get("embed") or {}
    media: list[dict] = []
    for img in embed.get("images") or []:
        media.append({
            "type": "image",
            "url": img.get("fullsize") or img.get("thumb"),
            "description": img.get("alt"),
        })
    external = embed.get("external")
    if external:
        title = external.get("title") or ""
        desc = external.get("description") or ""
        media.append({
            "type": "link",
            "url": external.get("uri"),
            "description": (title + (" — " + desc if desc else "")).strip(" —"),
        })
    return media or None
