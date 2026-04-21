"""
api/colab_sync.py

Helpers для відправки датасетів у Colab через chunked HTTP upload.

Використання:
    from api.colab_sync import ensure_dataset_on_colab
    ensure_dataset_on_colab(dataset_id, folder)  # викликати при activate

Env vars:
    COLAB_NGROK_URL   — base URL Colab сервера
    IS_COLAB          — якщо не true, функції нічого не роблять (local mode)
"""
import io
import os
import logging
import zipfile
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Chunk size — 8 MB (ngrok free ліміт має спокійно пройти)
CHUNK_SIZE_BYTES = 8 * 1024 * 1024

# HTTP timeouts
CONNECT_TIMEOUT = 10
UPLOAD_TIMEOUT = 120       # per chunk
FINALIZE_TIMEOUT = 300     # extraction може довго йти


class ColabSyncError(Exception):
    pass


def _colab_base_url() -> str | None:
    """Return colab URL if IS_COLAB=true, else None (local mode)."""
    is_colab = os.getenv("IS_COLAB", "false").lower() in ("true", "1", "t")
    if not is_colab:
        return None
    url = os.getenv("COLAB_NGROK_URL", "").strip().rstrip("/")
    if not url:
        raise ColabSyncError("IS_COLAB=true але COLAB_NGROK_URL порожній")
    return url


def dataset_exists_on_colab(dataset_id: int | str) -> bool:
    """Check via /dataset_status чи dataset вже у Drive."""
    base = _colab_base_url()
    if base is None:
        return False  # local mode — nothing to check
    try:
        r = requests.get(
            f"{base}/dataset_status",
            params={"dataset_id": str(dataset_id)},
            timeout=CONNECT_TIMEOUT,
        )
        r.raise_for_status()
        return bool(r.json().get("exists"))
    except requests.exceptions.RequestException as e:
        logger.warning(f"dataset_status check failed: {e}")
        return False


def _build_zip_in_memory(dataset_folder: str) -> bytes:
    """Запакувати усі CSV з dataset folder у ZIP у пам'яті."""
    folder = Path(dataset_folder)
    if not folder.exists():
        raise ColabSyncError(f"Dataset folder не існує: {folder}")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for csv_file in folder.glob("*.csv"):
            # arcname без шляху — просто назва файлу
            zf.write(csv_file, arcname=csv_file.name)
            logger.info(f"  Added {csv_file.name} ({csv_file.stat().st_size / 1024 / 1024:.1f} MB)")

    zip_bytes = buf.getvalue()
    logger.info(f"ZIP built: {len(zip_bytes) / 1024 / 1024:.1f} MB")
    return zip_bytes


def _upload_chunks(base_url: str, dataset_id: str, zip_bytes: bytes) -> int:
    """
    Розбити zip_bytes на чанки, надіслати послідовно.
    Returns: кількість надісланих чанків.
    """
    total_size = len(zip_bytes)
    total_chunks = (total_size + CHUNK_SIZE_BYTES - 1) // CHUNK_SIZE_BYTES
    logger.info(f"Uploading {total_size / 1024 / 1024:.1f} MB у {total_chunks} чанках...")

    for idx in range(total_chunks):
        start = idx * CHUNK_SIZE_BYTES
        end = min(start + CHUNK_SIZE_BYTES, total_size)
        chunk = zip_bytes[start:end]

        files = {"chunk": (f"chunk_{idx}.bin", chunk, "application/octet-stream")}
        data = {
            "dataset_id": dataset_id,
            "chunk_idx": str(idx),
            "total_chunks": str(total_chunks),
        }

        try:
            r = requests.post(
                f"{base_url}/upload_chunk",
                files=files,
                data=data,
                timeout=UPLOAD_TIMEOUT,
            )
            r.raise_for_status()
            resp = r.json()
            logger.info(f"  Chunk {idx + 1}/{total_chunks}: received={resp.get('chunks_received')}")
        except requests.exceptions.RequestException as e:
            raise ColabSyncError(f"Chunk {idx} upload failed: {e}")

    return total_chunks


def _finalize_on_colab(base_url: str, dataset_id: str) -> dict[str, Any]:
    """Тригер розпакування у Drive."""
    logger.info(f"Finalizing upload for dataset_id={dataset_id}...")
    try:
        r = requests.post(
            f"{base_url}/upload_finalize",
            json={"dataset_id": dataset_id},
            timeout=FINALIZE_TIMEOUT,
        )
        r.raise_for_status()
        resp = r.json()
        if not resp.get("ok"):
            raise ColabSyncError(f"Finalize failed: {resp.get('error')}")
        logger.info(f"Dataset {dataset_id} extracted on Drive: {resp.get('target_dir')}")
        return resp
    except requests.exceptions.RequestException as e:
        raise ColabSyncError(f"Finalize request failed: {e}")


def ensure_dataset_on_colab(
    dataset_id: int | str,
    dataset_folder: str,
    force: bool = False,
) -> dict[str, Any]:
    """
    Головна функція: переконатися що dataset є у Colab Drive.

    - Якщо IS_COLAB не true — повертає {skipped: True, reason: "local mode"}
    - Якщо dataset вже є на Colab і force=False — {skipped: True, reason: "already exists"}
    - Інакше: zip → chunked upload → finalize

    Args:
        dataset_id: ID у БД
        dataset_folder: шлях до папки з CSV локально
        force: якщо True — перепакує навіть якщо вже є

    Returns: dict з інформацією про upload або skip.
    """
    base = _colab_base_url()
    if base is None:
        return {"skipped": True, "reason": "IS_COLAB не true, local mode"}

    dataset_id_str = str(dataset_id)

    # Check if already there
    if not force:
        if dataset_exists_on_colab(dataset_id_str):
            logger.info(f"Dataset {dataset_id} вже у Drive, skip upload")
            return {"skipped": True, "reason": "already exists on Colab"}

    # Build ZIP
    logger.info(f"Building ZIP for dataset_id={dataset_id} from {dataset_folder}")
    zip_bytes = _build_zip_in_memory(dataset_folder)

    # Upload
    total_chunks = _upload_chunks(base, dataset_id_str, zip_bytes)

    # Finalize
    finalize_result = _finalize_on_colab(base, dataset_id_str)

    return {
        "skipped": False,
        "dataset_id": dataset_id_str,
        "zip_size_mb": round(len(zip_bytes) / 1024 / 1024, 2),
        "chunks_sent": total_chunks,
        "extracted_files": finalize_result.get("files", []),
        "target_dir": finalize_result.get("target_dir"),
    }
