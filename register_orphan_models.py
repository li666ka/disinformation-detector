"""
Зареєструвати моделі що існують на диску але відсутні у БД (orphans).

ВАЖЛИВО:
- Працює якщо backend має ДОСТУП до файлів (локально або через mount).
- НЕ змінює існуючі records у БД.
- Для моделей з predictions.json — підвантажить також у predictions_json колонку.

USE:
  python register_orphan_models.py --dry-run                 # preview
  python register_orphan_models.py --user-id 1               # commit для user_1
  python register_orphan_models.py --models-root <path>      # override MODELS_ROOT
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from api.database import ModelRecord, SessionLocal


MODELS_ROOT_CANDIDATES = [
    Path("/content/drive/MyDrive/diploma_models"),
    Path(os.path.expanduser(
        "~/Library/CloudStorage/GoogleDrive-lizazhyvitskaya@gmail.com/"
        "Мій диск/diploma_models"
    )),
    Path(__file__).resolve().parent.parent / "diploma-ml-server" / "data" / "models",
    Path(__file__).resolve().parent / "models",
    Path("./models"),
]


def find_models_root(override: str | None = None) -> Path | None:
    if override:
        p = Path(override).expanduser()
        return p if p.exists() else None
    for p in MODELS_ROOT_CANDIDATES:
        if p.exists() and p.is_dir():
            return p
    return None


def detect_model_type(model_dir_name: str) -> tuple[str, str]:
    """Витягти (model_type, pipeline_type) з імені директорії."""
    name = model_dir_name.lower()
    if name.startswith("nb_aggregated"):
        return "nb", "aggregated"
    if name.startswith("nb_"):
        return "nb", "article"
    if name.startswith("distilbert_"):
        return "distilbert", "article"
    if name.startswith("deberta_"):
        return "deberta", "article"
    if name.startswith("gin_"):
        return "gin", "graph"
    if name.startswith("sage_"):
        return "sage", "graph"
    if name.startswith("gnn_"):
        return "gnn", "graph"
    return "unknown", "unknown"


def extract_splits_from_name(model_dir_name: str) -> str | None:
    low = model_dir_name.lower()
    if low.endswith("_cross") or "_cross_" in low:
        return "cross_domain"
    if low.endswith("_in") or "_in_" in low:
        return "in_domain"
    if "_mixed" in low:
        return "mixed"
    return None


def _build_compact_predictions(preds_file: Path, model_type: str,
                               splits_used: str | None) -> str | None:
    try:
        with open(preds_file) as f:
            data = json.load(f)
    except Exception as e:
        print(f"    ⚠️  failed reading {preds_file.name}: {e}")
        return None
    preds_list = data.get("predictions", [])
    if not preds_list:
        return None
    compact = {
        "article_ids": [str(p["article_id"]) for p in preds_list],
        "y_true": [int(p["y_true"]) for p in preds_list],
        "y_pred": [int(p["y_pred"]) for p in preds_list],
        "y_proba_fake": [float(p["y_proba_fake"]) for p in preds_list],
        "model_type": data.get("model_type", model_type),
        "splits_used": data.get("splits_used", splits_used or ""),
        "dataset_id": str(data.get("dataset_id", "")),
        "test_size": len(preds_list),
        "created_at": data.get("created_at", ""),
    }
    return json.dumps(compact)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", type=int, default=None,
                        help="фільтр по user_X (за замовч. — всі user_*)")
    parser.add_argument("--models-root", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    models_root = find_models_root(args.models_root)
    if not models_root:
        print("❌ MODELS_ROOT не знайдено. Спробуй --models-root <path>.")
        return 1
    print(f"📂 MODELS_ROOT: {models_root}")
    print(f"🔧 Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")

    db = SessionLocal()
    registered_paths = {m.model_path for m in db.query(ModelRecord).all() if m.model_path}

    user_dirs = (
        [models_root / f"user_{args.user_id}"] if args.user_id is not None
        else sorted(models_root.glob("user_*"))
    )

    registered_count = 0
    skipped_unknown = 0

    for user_dir in user_dirs:
        if not user_dir.is_dir():
            continue
        print(f"\n• {user_dir.name}/")
        for model_dir in sorted(user_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_file: Path | None = None
            for candidate in ("model.pkl", "best_model.pt"):
                p = model_dir / candidate
                if p.exists():
                    model_file = p
                    break
            if not model_file:
                continue

            if str(model_file) in registered_paths:
                continue

            model_type, pipeline_type = detect_model_type(model_dir.name)
            if model_type == "unknown":
                print(f"  ? UNKNOWN type for {model_dir.name} — пропускаю")
                skipped_unknown += 1
                continue

            splits_used = extract_splits_from_name(model_dir.name)
            experiment_id = model_dir.name

            preds_file = model_dir / "predictions.json"
            predictions_json = None
            if preds_file.exists():
                predictions_json = _build_compact_predictions(
                    preds_file, model_type, splits_used
                )

            # Унікальний filename: <experiment_id>_model_<ts>
            timestamp = int(time.time() * 1000)
            filename = f"{experiment_id}_model_{timestamp}"
            while db.query(ModelRecord).filter(ModelRecord.filename == filename).first():
                timestamp += 1
                filename = f"{experiment_id}_model_{timestamp}"

            print(f"  {'WOULD' if args.dry_run else 'REGISTER'}: {model_dir.name}  "
                  f"({model_type}/{pipeline_type}, splits={splits_used or '—'}, "
                  f"preds={'✓' if predictions_json else '✗'})")

            if not args.dry_run:
                record = ModelRecord(
                    experiment_id=experiment_id,
                    filename=filename,
                    name=f"{model_type.upper()} {experiment_id}",
                    model_path=str(model_file),
                    model_type=model_type,
                    pipeline_type=pipeline_type,
                    metrics_json=None,
                    predictions_json=predictions_json,
                    splits_used=splits_used,
                    dataset_id=None,
                    is_active=False,
                    created_at=datetime.now(timezone.utc),
                )
                db.add(record)
                db.commit()
                registered_count += 1

    db.close()

    verb = "WOULD register" if args.dry_run else "Registered"
    print(f"\n{verb}: {registered_count} моделей")
    if skipped_unknown:
        print(f"Skipped (unknown type): {skipped_unknown}")
    if args.dry_run:
        print(f"\n💡 Запусти без --dry-run щоб виконати.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
