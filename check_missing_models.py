"""
Діагностика: знайти моделі на диску які НЕ зареєстровані у БД.

USE:
  python check_missing_models.py [--models-root <path>]
"""
import argparse
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from api.database import ModelRecord, SessionLocal


# Можливі MODELS_ROOT — як у migrate_models_to_subdirs.py.
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", type=str, default=None)
    args = parser.parse_args()

    models_root = find_models_root(args.models_root)
    if not models_root:
        print("❌ MODELS_ROOT не знайдено. Спробуй --models-root <path>.")
        for p in MODELS_ROOT_CANDIDATES:
            print(f"   tried: {p}")
        return 1
    print(f"📂 MODELS_ROOT: {models_root}")

    db = SessionLocal()
    all_records = db.query(ModelRecord).all()
    registered = {m.model_path for m in all_records if m.model_path}
    print(f"📊 Моделей у БД: {len(all_records)}")

    # Сканування файлів моделей у user_*/<type>_<exp>/
    found_files: list[dict] = []
    for user_dir in sorted(models_root.glob("user_*")):
        if not user_dir.is_dir():
            continue
        for model_dir in sorted(user_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for filename in ("model.pkl", "best_model.pt"):
                model_file = model_dir / filename
                if model_file.exists():
                    found_files.append({
                        "path": str(model_file),
                        "user": user_dir.name,
                        "model_dir": model_dir.name,
                        "size_kb": model_file.stat().st_size / 1024,
                        "has_predictions": (model_dir / "predictions.json").exists(),
                    })

    print(f"📦 Файлів моделей на диску: {len(found_files)}")

    # Orphans = на диску, але model_path не у БД.
    orphans = [f for f in found_files if f["path"] not in registered]
    if not orphans:
        print("\n✅ Всі файли зареєстровані у БД")
    else:
        print(f"\n⚠️  Orphan моделі (на диску, але не у БД): {len(orphans)}")
        print("─" * 70)
        for o in orphans:
            preds_mark = "✓ preds" if o["has_predictions"] else "✗ preds"
            print(f"  {o['user']}/{o['model_dir']}/ ({o['size_kb']:.0f} KB) [{preds_mark}]")
            print(f"    {o['path']}")

    # Дублікати filename у БД (UNIQUE constraint мав би їх відсікти, але про
    # всяк випадок).
    print("\n🔎 Перевірка дублікатів filename у БД:")
    filenames = [m.filename for m in all_records if m.filename]
    counter = Counter(filenames)
    duplicates = {fn: c for fn, c in counter.items() if c > 1}
    if duplicates:
        print(f"  ⚠️  Дублікати знайдено: {duplicates}")
    else:
        print(f"  ✅ Дублікатів немає ({len(filenames)} унікальних filename)")

    # Records у БД, але без model_path або без model_path на диску.
    missing_paths = [
        m for m in all_records
        if m.model_path and not Path(m.model_path).exists()
    ]
    if missing_paths:
        print(f"\n⚠️  Records у БД, чий model_path не знайдено на диску: {len(missing_paths)}")
        for m in missing_paths[:10]:
            print(f"  id={m.id} type={m.model_type} → {m.model_path}")
        if len(missing_paths) > 10:
            print(f"  ... ще {len(missing_paths) - 10}")

    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
