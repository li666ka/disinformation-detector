"""Health-aware client до Colab ML server.

Why:
  ngrok-tunnel часто падає (Colab перезапуск, ngrok session expire). Без
  health-check FE отримує HTML ngrok-помилки під виглядом 200/502/504 і
  показує їх як ML-failure. Цей модуль:
    - кешує перевірку 30s, щоб не бити Colab перед кожним /analyze
    - явно ловить ngrok HTML response (content-type text/html або
      ERR_NGROK у тілі) і трактує як offline
    - дає типізовані exceptions, які FastAPI handler конвертує у
      structured 503

Env:
  ML_SERVER_URL  — preferred (новий)
  COLAB_NGROK_URL — legacy fallback (зберігаємо для існуючих .env)
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

HEALTH_PATH = "/health"
HEALTH_TIMEOUT = 3.0
CACHE_TTL = 30.0

_cache_lock = threading.Lock()
_cache: dict = {"checked_at": 0.0, "result": None, "url": None}


class MLServerError(RuntimeError):
    """Базова. status_code підказує HTTP-код для FastAPI handler."""
    status_code: int = 503

    def __init__(self, message: str, checked_url: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.checked_url = checked_url


class MLServerOfflineError(MLServerError):
    """Tunnel впав / URL не виставлений / DNS fail / connection refused / ngrok HTML."""


class MLServerNotReadyError(MLServerError):
    """HTTP 200, але /health повернув не "ok" (моделі ще завантажуються тощо)."""


def colab_url() -> Optional[str]:
    """Resolved URL з env. None якщо не виставлено в жодній змінній."""
    url = (
        os.environ.get("ML_SERVER_URL", "").strip()
        or os.environ.get("COLAB_NGROK_URL", "").strip()
    )
    return url.rstrip("/") if url else None


def _is_ngrok_error_page(response: requests.Response) -> bool:
    """Розпізнати ngrok error page під виглядом HTTP-відповіді."""
    ct = (response.headers.get("content-type") or "").lower()
    if "text/html" in ct:
        return True
    body_head = response.text[:2000] if response.text else ""
    return "ERR_NGROK" in body_head or "ngrok-free.app" in body_head and "<html" in body_head.lower()


def _probe(url: str) -> dict:
    """Один реальний HTTP-запит до /health. Без кешу, без exceptions —
    повертає dict-результат який потім кешується/перетворюється."""
    target = f"{url}{HEALTH_PATH}"
    logger.debug(f"ml_client probe → {target}")
    try:
        r = requests.get(target, timeout=HEALTH_TIMEOUT)
    except requests.RequestException as e:
        return {
            "ok": False,
            "reachable": False,
            "ready": False,
            "detail": f"{type(e).__name__}: {e}",
        }

    if _is_ngrok_error_page(r):
        return {
            "ok": False,
            "reachable": False,
            "ready": False,
            "detail": "ngrok tunnel offline (HTML error page)",
        }

    if r.status_code != 200:
        return {
            "ok": False,
            "reachable": True,
            "ready": False,
            "detail": f"HTTP {r.status_code}",
        }

    try:
        data = r.json()
    except ValueError:
        return {
            "ok": False,
            "reachable": True,
            "ready": False,
            "detail": "non-JSON /health response",
        }

    status = str(data.get("status") or "").lower()
    if status != "ok":
        return {
            "ok": False,
            "reachable": True,
            "ready": False,
            "detail": f"/health status={status!r}",
            "raw": data,
        }

    return {
        "ok": True,
        "reachable": True,
        "ready": True,
        "detail": "ML server available",
        "raw": data,
    }


def check_status(force: bool = False) -> dict:
    """Cached health check. Повертає dict — НЕ raise.

    Структура:
      {
        "ok": bool,
        "url_set": bool,
        "checked_url": str | None,
        "reachable": bool,
        "ready": bool,
        "detail": str,
        "cached": bool,
        "checked_at": float,  # unix ts
      }
    """
    url = colab_url()
    if not url:
        return {
            "ok": False,
            "url_set": False,
            "checked_url": None,
            "reachable": False,
            "ready": False,
            "detail": (
                "ML_SERVER_URL (або COLAB_NGROK_URL) не встановлено — "
                "DistilBERT/GNN/aggregated-NB недоступні"
            ),
            "cached": False,
            "checked_at": time.time(),
        }

    now = time.time()
    with _cache_lock:
        cached = _cache["result"]
        fresh = (
            not force
            and cached is not None
            and _cache["url"] == url
            and (now - _cache["checked_at"]) < CACHE_TTL
        )
        if fresh:
            return {**cached, "cached": True}

    probe_result = _probe(url)
    result = {
        "url_set": True,
        "checked_url": url,
        "cached": False,
        "checked_at": now,
        **probe_result,
    }
    with _cache_lock:
        _cache["checked_at"] = now
        _cache["url"] = url
        _cache["result"] = result
    return result


def invalidate_cache() -> None:
    with _cache_lock:
        _cache["checked_at"] = 0.0
        _cache["result"] = None
        _cache["url"] = None


def ensure_healthy() -> str:
    """Перед кожним викликом Colab. Повертає валідний URL або raise.

    Raises:
      MLServerOfflineError — URL не виставлений / unreachable / ngrok HTML
      MLServerNotReadyError — reachable, але /health != "ok"
    """
    status = check_status()
    url = status.get("checked_url")

    if not status["url_set"]:
        raise MLServerOfflineError(status["detail"], checked_url=None)

    if not status["reachable"]:
        raise MLServerOfflineError(status["detail"], checked_url=url)

    if not status["ready"]:
        raise MLServerNotReadyError(status["detail"], checked_url=url)

    return url
