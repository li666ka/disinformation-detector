"""
Dataset service — handles ZIP upload, validation, storage, and statistics.

ZIP must contain at least `news.csv` with columns (article_id, article_text, article_label).
Optionally may contain: tweets.csv, retweets.csv, replies.csv, likes.csv,
users.csv, evidence.csv with appropriate columns and FK references.
"""
from __future__ import annotations
import io
import json
import logging
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────

# Project root / datasets folder
BASE_PATH = Path(__file__).parent.parent.resolve()
DATASETS_ROOT = BASE_PATH / "uploaded_datasets"

# Upload limits
MAX_ZIP_SIZE_MB = 500
MAX_ZIP_SIZE_BYTES = MAX_ZIP_SIZE_MB * 1024 * 1024
MAX_NEWS_ROWS = 100_000

# Required/optional files in a dataset
REQUIRED_FILES = ["news.csv"]
OPTIONAL_FILES = ["tweets.csv", "retweets.csv", "replies.csv", "likes.csv", "users.csv", "evidence.csv"]
ALL_FILES = REQUIRED_FILES + OPTIONAL_FILES

# Required columns for each file
REQUIRED_COLUMNS = {
    "news.csv":     ["article_id", "article_text", "article_label"],
    "tweets.csv":   ["tweet_id", "article_id"],
    "retweets.csv": ["original_tweet_id", "user_id"],
    "replies.csv":  ["reply_id", "parent_tweet_id"],
    "likes.csv":    ["tweet_id", "user_id"],
    "users.csv":    ["user_id"],
    "evidence.csv": ["article_id"],
}

VALID_LABELS = {"FAKE", "REAL"}

# Safe filename pattern (prevent path traversal)
SAFE_FILENAME_RE = re.compile(r"^[a-zA-Z0-9_.-]+$")


# ── Exceptions ───────────────────────────────────────────────────────────

class DatasetValidationError(Exception):
    """Raised when uploaded ZIP fails validation."""
    pass


# ── Storage path helpers ─────────────────────────────────────────────────

def get_dataset_folder(user_id: int, dataset_id: int) -> Path:
    """Return absolute folder path for a given dataset."""
    folder = DATASETS_ROOT / f"user_{user_id}" / f"dataset_{dataset_id}"
    return folder


def ensure_user_folder(user_id: int) -> Path:
    user_folder = DATASETS_ROOT / f"user_{user_id}"
    user_folder.mkdir(parents=True, exist_ok=True)
    return user_folder


# ── ZIP validation ───────────────────────────────────────────────────────

def validate_zip_structure(zip_bytes: bytes) -> tuple[zipfile.ZipFile, dict[str, str]]:
    """
    Open ZIP from bytes, verify basic structure. Returns the ZipFile and a
    mapping {csv_name: internal_path_in_zip} for found files.

    Raises DatasetValidationError on any structural issue.
    """
    if len(zip_bytes) > MAX_ZIP_SIZE_BYTES:
        raise DatasetValidationError(
            f"ZIP завеликий: {len(zip_bytes) / 1024 / 1024:.1f} MB "
            f"(максимум {MAX_ZIP_SIZE_MB} MB)"
        )

    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile:
        raise DatasetValidationError("Файл не є валідним ZIP архівом")

    # Build index {basename: internal_path}, skipping directories and hidden files
    # This handles both `news.csv` at root AND `my-dataset/news.csv` inside a folder
    found: dict[str, str] = {}
    for info in zf.infolist():
        if info.is_dir():
            continue
        name = os.path.basename(info.filename)
        if not name or name.startswith(".") or name.startswith("__"):
            continue
        if name.lower() in ALL_FILES:
            key = name.lower()
            if key in found:
                raise DatasetValidationError(
                    f"У ZIP знайдено кілька файлів {key} — має бути один"
                )
            # Path traversal protection
            if ".." in info.filename or info.filename.startswith("/"):
                raise DatasetValidationError(f"Небезпечний шлях у ZIP: {info.filename}")
            found[key] = info.filename

    # Check required files
    for req in REQUIRED_FILES:
        if req not in found:
            raise DatasetValidationError(
                f"У ZIP не знайдено обов'язковий файл: {req}"
            )

    return zf, found


def _read_csv_from_zip(zf: zipfile.ZipFile, internal_path: str) -> pd.DataFrame:
    """Read a CSV file from an open ZIP archive."""
    with zf.open(internal_path) as f:
        return pd.read_csv(f, dtype=str, keep_default_na=False, na_values=[""])


def validate_csv_schemas(zf: zipfile.ZipFile, found: dict[str, str]) -> tuple[dict, list[str]]:
    """
    Validate each CSV's schema (required columns, types, FK references).

    Returns (dataframes_by_name, warnings).
    Raises DatasetValidationError for fatal issues.
    """
    dataframes: dict[str, pd.DataFrame] = {}
    warnings: list[str] = []

    # Read all CSVs
    for name, internal_path in found.items():
        try:
            df = _read_csv_from_zip(zf, internal_path)
        except Exception as e:
            raise DatasetValidationError(f"Не вдалося прочитати {name}: {e}")

        # Check required columns
        required = REQUIRED_COLUMNS.get(name, [])
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise DatasetValidationError(
                f"У файлі {name} відсутні обов'язкові колонки: {missing}"
            )

        dataframes[name] = df

    # Validate news.csv content
    news_df = dataframes["news.csv"]
    if len(news_df) == 0:
        raise DatasetValidationError("news.csv порожній")
    if len(news_df) > MAX_NEWS_ROWS:
        raise DatasetValidationError(
            f"news.csv має {len(news_df)} рядків — максимум {MAX_NEWS_ROWS}"
        )

    # Validate article_label column: must be FAKE, REAL, or empty
    if "article_label" in news_df.columns:
        def check_label(v):
            if pd.isna(v) or v == "":
                return True
            return str(v).strip().upper() in VALID_LABELS

        invalid_labels = news_df[~news_df["article_label"].apply(check_label)]
        if len(invalid_labels) > 0:
            examples = invalid_labels["article_label"].head(3).tolist()
            raise DatasetValidationError(
                f"У news.csv знайдено {len(invalid_labels)} рядків з невалідним article_label. "
                f"Приклади: {examples}. Допустимі значення: FAKE, REAL, або порожньо."
            )

    # Check duplicate article_id
    if news_df["article_id"].duplicated().any():
        dupes = news_df["article_id"].duplicated().sum()
        warnings.append(f"news.csv має {dupes} дублікатів article_id")

    # Check empty article_text
    empty_texts = news_df["article_text"].apply(lambda x: not str(x).strip()).sum()
    if empty_texts > 0:
        warnings.append(f"news.csv має {empty_texts} рядків з порожнім article_text")

    # Validate FK relationships if tweets.csv is present
    if "tweets.csv" in dataframes:
        article_ids = set(news_df["article_id"].astype(str))
        tweets_df = dataframes["tweets.csv"]
        orphan_tweets = tweets_df[~tweets_df["article_id"].astype(str).isin(article_ids)]
        if len(orphan_tweets) > 0:
            warnings.append(
                f"tweets.csv має {len(orphan_tweets)} рядків з article_id, якого немає в news.csv"
            )

    # Validate FK: replies → tweets
    if "replies.csv" in dataframes and "tweets.csv" in dataframes:
        tweet_ids = set(dataframes["tweets.csv"]["tweet_id"].astype(str))
        orphan_replies = dataframes["replies.csv"][
            ~dataframes["replies.csv"]["parent_tweet_id"].astype(str).isin(tweet_ids)
        ]
        if len(orphan_replies) > 0:
            warnings.append(
                f"replies.csv має {len(orphan_replies)} рядків з невідомим parent_tweet_id"
            )

    return dataframes, warnings


# ── Storage operations ───────────────────────────────────────────────────

def save_dataset_files(
    user_id: int,
    dataset_id: int,
    dataframes: dict[str, pd.DataFrame],
    name: str,
    description: Optional[str],
) -> Path:
    """
    Persist dataset to disk. Creates folder structure:
      uploaded_datasets/user_<user_id>/dataset_<id>/
        ├── manifest.json
        ├── news.csv
        └── [other csvs]
    Returns the dataset folder path.
    """
    folder = get_dataset_folder(user_id, dataset_id)
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)

    # Save each CSV
    for filename, df in dataframes.items():
        out_path = folder / filename
        df.to_csv(out_path, index=False)

    # Save manifest
    manifest = {
        "name": name,
        "description": description,
        "components": list(dataframes.keys()),
        "row_counts": {name: int(len(df)) for name, df in dataframes.items()},
    }
    (folder / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return folder


def delete_dataset_folder(folder_path: str) -> None:
    """Remove dataset folder from disk. Idempotent."""
    p = Path(folder_path)
    if p.exists() and p.is_dir():
        shutil.rmtree(p)


def compute_folder_size(folder: Path) -> int:
    """Sum of all file sizes in folder (bytes)."""
    total = 0
    for path in folder.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


# ── Statistics ───────────────────────────────────────────────────────────

def compute_basic_stats(news_df: pd.DataFrame) -> dict:
    """Compute fake/real/unlabeled counts from news.csv."""
    total = len(news_df)
    fake = 0
    real = 0
    unlabeled = 0

    for v in news_df["article_label"]:
        if pd.isna(v) or v == "":
            unlabeled += 1
            continue
        label = str(v).strip().upper()
        if label == "FAKE":
            fake += 1
        elif label == "REAL":
            real += 1
        else:
            unlabeled += 1

    return {
        "total": total,
        "fake": fake,
        "real": real,
        "unlabeled": unlabeled,
    }


def get_preview(dataframes: dict[str, pd.DataFrame], n_rows: int = 5) -> dict:
    """Return first N rows of news.csv + row counts for components."""
    news_df = dataframes["news.csv"]
    preview_df = news_df.head(n_rows)

    # Convert to list of dicts, truncating long text
    news_head = []
    for _, row in preview_df.iterrows():
        item = {}
        for col in row.index:
            val = row[col]
            if pd.isna(val):
                item[col] = None
            else:
                str_val = str(val)
                if col == "article_text" and len(str_val) > 200:
                    str_val = str_val[:200] + "…"
                item[col] = str_val
        news_head.append(item)

    counts = {name: int(len(df)) for name, df in dataframes.items()}

    return {
        "news_head": news_head,
        "counts": counts,
        "news_columns": list(news_df.columns),
    }


def compute_detailed_stats(folder_path: str) -> dict:
    """
    Deep statistics for /datasets/{id}/stats endpoint.
    Reads CSVs from disk and computes aggregates.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Dataset folder not found: {folder_path}")

    result: dict = {}

    # news.csv
    news_path = folder / "news.csv"
    if not news_path.exists():
        raise FileNotFoundError("news.csv missing in dataset folder")
    news_df = pd.read_csv(news_path, dtype=str, keep_default_na=False, na_values=[""])

    basic = compute_basic_stats(news_df)
    result["total_news"] = basic["total"]
    result["fake_count"] = basic["fake"]
    result["real_count"] = basic["real"]
    result["unlabeled_count"] = basic["unlabeled"]

    # Text length stats
    text_lens = news_df["article_text"].fillna("").astype(str).str.len()
    result["text_length_stats"] = {
        "min": int(text_lens.min()) if len(text_lens) else 0,
        "median": float(text_lens.median()) if len(text_lens) else 0,
        "mean": float(text_lens.mean()) if len(text_lens) else 0,
        "max": int(text_lens.max()) if len(text_lens) else 0,
        "p95": float(text_lens.quantile(0.95)) if len(text_lens) else 0,
    }
    result["empty_text_count"] = int((text_lens == 0).sum())
    result["duplicate_text_count"] = int(news_df["article_text"].duplicated().sum())

    # Top domains (if domain column present)
    top_domains = []
    if "domain" in news_df.columns:
        domain_counts = news_df["domain"].fillna("").value_counts().head(10)
        top_domains = [
            {"domain": str(d) if d else "(unknown)", "count": int(c)}
            for d, c in domain_counts.items()
        ]
    result["top_domains"] = top_domains

    # Tweets stats
    tweets_path = folder / "tweets.csv"
    if tweets_path.exists():
        tweets_df = pd.read_csv(tweets_path, dtype=str, keep_default_na=False, na_values=[""])
        result["total_tweets"] = int(len(tweets_df))
        if len(tweets_df) > 0 and "article_id" in tweets_df.columns:
            tweets_per_news = tweets_df.groupby("article_id").size()
            result["avg_tweets_per_news"] = float(tweets_per_news.mean())
            news_with_tweets = tweets_df["article_id"].nunique()
            if basic["total"] > 0:
                result["news_with_tweets_pct"] = float(news_with_tweets / basic["total"] * 100)

    # Engagement
    likes_path = folder / "likes.csv"
    if likes_path.exists():
        result["total_likes"] = int(len(pd.read_csv(likes_path)))

    retweets_path = folder / "retweets.csv"
    if retweets_path.exists():
        result["total_retweets"] = int(len(pd.read_csv(retweets_path)))

    replies_path = folder / "replies.csv"
    if replies_path.exists():
        result["total_replies"] = int(len(pd.read_csv(replies_path)))

    # Users
    users_path = folder / "users.csv"
    if users_path.exists():
        users_df = pd.read_csv(users_path, dtype=str, keep_default_na=False, na_values=[""])
        result["total_users"] = int(len(users_df))

        if "verified" in users_df.columns and len(users_df) > 0:
            def to_bool(v):
                if pd.isna(v) or v == "":
                    return False
                s = str(v).lower()
                return s in ("true", "1", "yes", "t")
            verified = users_df["verified"].apply(to_bool).sum()
            result["verified_users_pct"] = float(verified / len(users_df) * 100)

        if "followers_count" in users_df.columns and len(users_df) > 0:
            def to_int(v):
                try:
                    return int(float(v)) if v and not pd.isna(v) else 0
                except (ValueError, TypeError):
                    return 0
            followers = users_df["followers_count"].apply(to_int)
            result["avg_followers_count"] = float(followers.mean())

    return result


# ── Component detection ──────────────────────────────────────────────────

def detect_components(dataframes: dict[str, pd.DataFrame]) -> dict[str, bool]:
    """Return {component_name: bool} indicating which CSVs are non-empty."""
    return {
        "has_news": "news.csv" in dataframes and len(dataframes["news.csv"]) > 0,
        "has_tweets": "tweets.csv" in dataframes and len(dataframes["tweets.csv"]) > 0,
        "has_retweets": "retweets.csv" in dataframes and len(dataframes["retweets.csv"]) > 0,
        "has_replies": "replies.csv" in dataframes and len(dataframes["replies.csv"]) > 0,
        "has_likes": "likes.csv" in dataframes and len(dataframes["likes.csv"]) > 0,
        "has_users": "users.csv" in dataframes and len(dataframes["users.csv"]) > 0,
        "has_evidence": "evidence.csv" in dataframes and len(dataframes["evidence.csv"]) > 0,
    }
