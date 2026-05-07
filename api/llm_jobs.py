"""In-memory tracking для LLM evaluation jobs.

LLM evaluation повільна (5-10 хв на 100 семплів), тому endpoint
повертає job_id і UI робить polling /models/llm-jobs/{job_id}.
"""
from __future__ import annotations

import threading
import uuid
from typing import Optional

# {job_id: {"status": "pending"|"running"|"done"|"failed",
#           "progress": (current, total),
#           "result": dict | None, "error": str | None}}
_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def create_job() -> str:
    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = {
            "status": "pending",
            "progress": (0, 0),
            "result": None,
            "error": None,
        }
    return job_id


def update_progress(job_id: str, current: int, total: int) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id]["progress"] = (current, total)
            _jobs[job_id]["status"] = "running"


def mark_done(job_id: str, result: dict) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["result"] = result


def mark_failed(job_id: str, error: str) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = error


def get_status(job_id: str) -> Optional[dict]:
    with _lock:
        if job_id not in _jobs:
            return None
        return dict(_jobs[job_id])  # shallow copy
