"""
api/routers/datasets.py — CRUD endpoints for user datasets.

Endpoints:
  POST   /datasets/upload         — upload ZIP, validate, save, return preview
  GET    /datasets                — list user's datasets
  GET    /datasets/{id}           — dataset details
  GET    /datasets/{id}/stats     — detailed statistics (computed on demand)
  GET    /datasets/{id}/template  — download sample ZIP template
  POST   /datasets/{id}/activate  — mark as active (deactivates others)
  PATCH  /datasets/{id}           — update name/description
  DELETE /datasets/{id}           — remove from DB + filesystem
  GET    /datasets/active/info    — return currently active dataset (for training)
"""
import io
import logging
import os
import zipfile
from pathlib import Path
import requests
from fastapi import APIRouter, Body, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from api.database import get_db, Dataset, User
from api.auth import get_current_user
from api.colab_sync import ensure_dataset_on_colab, ColabSyncError
from api.dataset_service import (
    validate_zip_structure,
    validate_csv_schemas,
    save_dataset_files,
    save_dataset_splits,
    delete_dataset_folder,
    compute_folder_size,
    compute_basic_stats,
    compute_detailed_stats,
    get_preview,
    detect_components,
    get_dataset_folder,
    DatasetValidationError,
    MAX_ZIP_SIZE_MB,
)
from api.schemas_dataset import (
    DatasetResponse,
    DatasetUploadResponse,
    DatasetStatsResponse,
    DatasetUpdate,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["datasets"])


# ── POST /datasets/upload ────────────────────────────────────────────────

@router.post("/upload", response_model=DatasetUploadResponse, status_code=201)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(..., min_length=1, max_length=200),
    description: str = Form("", max_length=2000),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Upload a ZIP archive containing CSV files (news.csv + optional others).
    """
    # Check filename extension
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Очікується ZIP файл (.zip)")

    # Check for duplicate name (per user)
    existing = db.query(Dataset).filter(
        Dataset.user_id == current_user.id,
        Dataset.name == name,
    ).first()
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Датасет з назвою '{name}' вже існує у вас",
        )

    # Read bytes
    zip_bytes = await file.read()

    # Validate ZIP structure
    try:
        zf, found_files, splits_in_zip = validate_zip_structure(zip_bytes)
    except DatasetValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Validate schemas + content. We need the ZipFile open later for splits, тому
    # читаємо CSVs у dataframes але не закриваємо archive (не використовуємо `with zf`).
    try:
        dataframes, warnings = validate_csv_schemas(zf, found_files)
    except DatasetValidationError as e:
        zf.close()
        raise HTTPException(status_code=400, detail=str(e))

    # Compute basic stats
    news_df = dataframes["news.csv"]
    basic_stats = compute_basic_stats(news_df)
    components = detect_components(dataframes)

    # Create DB record first (to get ID for folder path)
    record = Dataset(
        user_id=current_user.id,
        name=name,
        description=description.strip() or None,
        folder_path="",  # filled after save
        total_news=basic_stats["total"],
        fake_count=basic_stats["fake"],
        real_count=basic_stats["real"],
        unlabeled_count=basic_stats["unlabeled"],
        **components,
        file_size_bytes=0,
        is_active=False,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    # Save files to disk (now we have dataset_id)
    try:
        folder = save_dataset_files(
            current_user.id, record.id, dataframes,
            name=name, description=description,
        )
        saved_splits = save_dataset_splits(folder, zf, splits_in_zip)
    except Exception as e:
        db.delete(record)
        db.commit()
        zf.close()
        logger.exception("Failed to save dataset files")
        raise HTTPException(status_code=500, detail=f"Не вдалося зберегти файли: {e}")
    finally:
        # Усі читання з ZIP завершено — закриваємо archive.
        try:
            zf.close()
        except Exception:
            pass

    if saved_splits:
        warnings.append(
            f"Знайдено фіксовані splits: {', '.join(sorted(saved_splits))}"
        )
        logger.info(f"Dataset id={record.id} saved with splits: {saved_splits}")

    # Update record with folder path and size
    record.folder_path = str(folder)
    record.file_size_bytes = compute_folder_size(folder)
    db.commit()
    db.refresh(record)

    # Preview
    preview = get_preview(dataframes, n_rows=5)

    logger.info(
        f"Uploaded dataset id={record.id}, user={current_user.id}, "
        f"name={name}, news_rows={basic_stats['total']}"
    )
    return DatasetUploadResponse(
        dataset=DatasetResponse.model_validate(record),
        preview=preview,
        warnings=warnings,
    )


# ── GET /datasets ────────────────────────────────────────────────────────

@router.get("", response_model=list[DatasetResponse])
def list_datasets(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all datasets of current user, newest first."""
    records = (
        db.query(Dataset)
        .filter(Dataset.user_id == current_user.id)
        .order_by(Dataset.created_at.desc())
        .all()
    )
    return records


# ── GET /datasets/{id} ───────────────────────────────────────────────────

@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    record = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        Dataset.user_id == current_user.id,
    ).first()
    if not record:
        raise HTTPException(status_code=404, detail="Датасет не знайдено")
    return record


# ── GET /datasets/{id}/stats ─────────────────────────────────────────────

@router.get("/{dataset_id}/stats", response_model=DatasetStatsResponse)
def get_dataset_stats(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Detailed statistics — computed on demand from CSV files."""
    record = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        Dataset.user_id == current_user.id,
    ).first()
    if not record:
        raise HTTPException(status_code=404, detail="Датасет не знайдено")

    try:
        stats = compute_detailed_stats(record.folder_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Файли датасету не знайдено: {e}")
    except Exception as e:
        logger.exception("Failed to compute stats")
        raise HTTPException(status_code=500, detail=f"Помилка обчислення статистики: {e}")

    return DatasetStatsResponse(dataset_id=dataset_id, **stats)


# ── GET /datasets/{id}/splits ────────────────────────────────────────────

def _count_csv_rows(path) -> int:
    """Кількість data-рядків у CSV (без header). Швидкий побайтовий count."""
    try:
        with open(path, "rb") as f:
            n = sum(1 for _ in f)
        return max(0, n - 1)
    except OSError:
        return 0


@router.get("/{dataset_id}/splits")
def get_dataset_splits(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List splits_<name>/ subfolders на локальному дисковому датасеті.

    Source of truth — upload (`save_dataset_splits`). На Colab вони з'являються
    лише після /activate, тому ходити туди за списком немає сенсу.
    """
    ds = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        Dataset.user_id == current_user.id,
    ).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Датасет не знайдено")

    folder = Path(ds.folder_path) if ds.folder_path else None
    splits: list[dict] = []
    if folder and folder.exists():
        for sub in sorted(folder.iterdir()):
            if not sub.is_dir() or not sub.name.startswith("splits_"):
                continue
            train = sub / "split_train.csv"
            val = sub / "split_val.csv"
            test = sub / "split_test.csv"
            if not (train.exists() and val.exists() and test.exists()):
                continue
            splits.append({
                "name": sub.name[len("splits_"):],
                "folder": sub.name,
                "train_count": _count_csv_rows(train),
                "val_count": _count_csv_rows(val),
                "test_count": _count_csv_rows(test),
            })

    return {
        "dataset_id": dataset_id,
        "splits": splits,
        "has_legacy_splits": False,
    }


# ── PATCH /datasets/{id}/active-split ────────────────────────────────────

@router.patch("/{dataset_id}/active-split")
def set_active_split(
    dataset_id: int,
    payload: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Set active split name (or null for auto-split)."""
    ds = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        Dataset.user_id == current_user.id,
    ).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Датасет не знайдено")

    split_name = payload.get("split_name")
    if split_name is not None and not isinstance(split_name, str):
        raise HTTPException(status_code=400, detail="split_name must be string or null")

    ds.active_split = split_name or None
    db.commit()
    db.refresh(ds)

    return {"id": ds.id, "active_split": ds.active_split}


# ── GET /datasets/{id}/template ──────────────────────────────────────────

@router.get("/template/download")
def download_template(_: User = Depends(get_current_user)):
    """Download a sample ZIP template showing the expected structure."""
    import pandas as pd

    # Build minimal example CSVs in memory
    news = pd.DataFrame([
        {"article_id": "article_001", "article_text": "Scientists discover water on Mars.", "article_label": "REAL",
         "title": "Mars Water Found", "url": "https://example.com/mars", "domain": "example.com"},
        {"article_id": "article_002", "article_text": "Celebrity secretly marries alien, report claims.", "article_label": "FAKE",
         "title": "Alien Wedding", "url": "https://tabloid.com/alien", "domain": "tabloid.com"},
    ])
    tweets = pd.DataFrame([
        {"tweet_id": "t1", "article_id": "article_001", "text": "Amazing discovery!",
         "user_id": "u1", "favourite_count": 10, "retweet_count": 3, "reply_count": 1},
        {"tweet_id": "t2", "article_id": "article_002", "text": "This is fake!",
         "user_id": "u2", "favourite_count": 5, "retweet_count": 0, "reply_count": 2},
    ])
    users = pd.DataFrame([
        {"user_id": "u1", "screen_name": "sciencefan", "followers_count": 1500,
         "friends_count": 200, "statuses_count": 5000, "verified": True,
         "created_at": "2015-03-20"},
        {"user_id": "u2", "screen_name": "skeptic99", "followers_count": 250,
         "friends_count": 180, "statuses_count": 1200, "verified": False,
         "created_at": "2020-07-15"},
    ])

    # Zip them up
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("news.csv", news.to_csv(index=False))
        zf.writestr("tweets.csv", tweets.to_csv(index=False))
        zf.writestr("users.csv", users.to_csv(index=False))

        # Also include a README
        readme = """Шаблон датасету для fake news detection

ОБОВ'ЯЗКОВИЙ ФАЙЛ:
  news.csv — колонки: article_id, article_text, article_label
    - article_id: унікальний ID (рядок)
    - article_text: текст новини
    - article_label: REAL, FAKE, або порожньо = нерозмічено
    - (опц.) title, url, domain, authors, publish_date

ОПЦІОНАЛЬНІ ФАЙЛИ:
  tweets.csv — tweet_id, article_id (FK), text, user_id (FK), favourite_count, retweet_count, reply_count
  retweets.csv — original_tweet_id (FK), user_id (FK)
  replies.csv — reply_id, parent_tweet_id (FK), parent_reply_id, user_id, text
  likes.csv — tweet_id (FK), user_id (FK)
  users.csv — user_id, screen_name, followers_count, friends_count, statuses_count, verified, created_at
  evidence.csv — article_id (FK), url, content

Усі CSV мають бути у корені ZIP архіву, або всередині однієї папки.
"""
        zf.writestr("README.txt", readme)

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="dataset_template.zip"'},
    )


# ── POST /datasets/{id}/activate ─────────────────────────────────────────

@router.post("/{dataset_id}/activate")
def activate_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Mark as active. Deactivates any other active dataset for this user."""
    record = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        Dataset.user_id == current_user.id,
    ).first()
    if not record:
        raise HTTPException(status_code=404, detail="Датасет не знайдено")

    # Deactivate all others for this user
    db.query(Dataset).filter(
        Dataset.user_id == current_user.id,
        Dataset.id != dataset_id,
    ).update({"is_active": False})

    record.is_active = True
    db.commit()
    db.refresh(record)

    try:
        sync_result = ensure_dataset_on_colab(
            dataset_id=dataset_id,
            dataset_folder=record.folder_path,
        )
        logger.info(f"Colab sync: {sync_result}")
    except ColabSyncError as e:
        logger.error(f"Colab sync failed: {e}")
        return {
            "ok": True,
            "activated": True,
            "colab_sync": {"ok": False, "error": str(e)},
        }

    return {
        "ok": True,
        "activated": True,
        "colab_sync": sync_result,
    }


# ── PATCH /datasets/{id} ─────────────────────────────────────────────────

@router.patch("/{dataset_id}", response_model=DatasetResponse)
def update_dataset(
    dataset_id: int,
    upd: DatasetUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    record = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        Dataset.user_id == current_user.id,
    ).first()
    if not record:
        raise HTTPException(status_code=404, detail="Датасет не знайдено")

    if upd.name is not None:
        # Check uniqueness within user
        other = db.query(Dataset).filter(
            Dataset.user_id == current_user.id,
            Dataset.name == upd.name,
            Dataset.id != dataset_id,
        ).first()
        if other:
            raise HTTPException(status_code=400, detail="Назва вже використовується")
        record.name = upd.name

    if upd.description is not None:
        record.description = upd.description.strip() or None

    db.commit()
    db.refresh(record)
    return record


# ── DELETE /datasets/{id} ────────────────────────────────────────────────

@router.delete("/{dataset_id}")
def delete_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    record = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        Dataset.user_id == current_user.id,
    ).first()
    if not record:
        raise HTTPException(status_code=404, detail="Датасет не знайдено")

    folder_path = record.folder_path
    db.delete(record)
    db.commit()

    # Remove files (best-effort)
    try:
        delete_dataset_folder(folder_path)
    except Exception as e:
        logger.warning(f"Could not delete folder {folder_path}: {e}")

    return {"ok": True, "deleted_id": dataset_id}


# ── GET /datasets/active/info ────────────────────────────────────────────

@router.get("/active/info", response_model=DatasetResponse | None)
def get_active_dataset(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return currently active dataset for training, or null."""
    record = db.query(Dataset).filter(
        Dataset.user_id == current_user.id,
        Dataset.is_active == True,
    ).first()
    return record
