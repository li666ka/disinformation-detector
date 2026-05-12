"""
One-time migration: переносить старі моделі у нову структуру.

Старе:
  user_X/model_xxx.pkl
  user_X/model_nb_xxx.pkl
  user_X/model_nb_aggregated.pkl
  user_X/predictions.json  (orphan)

Нове:
  user_X/nb_xxx/model.pkl
  user_X/nb_aggregated_xxx/model.pkl
  (distilbert/gnn bundle-pkl лишаються — вони лише вказівники на свій save_dir)

USE:
  python migrate_models_to_subdirs.py --dry-run    # перевірити що буде
  python migrate_models_to_subdirs.py              # виконати

Або у Colab:
  !python /content/diploma-ml-server/migrate_models_to_subdirs.py --dry-run
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

try:
    import joblib  # для peek у bundle
except ImportError:
    joblib = None


# Можливі шляхи MODELS_ROOT — спробуємо знайти існуючий
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


def find_models_root() -> Path | None:
    for p in MODELS_ROOT_CANDIDATES:
        if p.exists() and p.is_dir():
            return p
    return None


def _peek_bundle_type(pkl_file: Path) -> tuple[str | None, str | None]:
    """Повернути (type, pipeline_type) з bundle pkl якщо це dict.
    None якщо не вдалось прочитати."""
    if joblib is None:
        return None, None
    try:
        data = joblib.load(pkl_file)
    except Exception:
        return None, None
    if isinstance(data, dict):
        return data.get("type"), data.get("pipeline_type")
    return None, None


def _classify_pkl(pkl_file: Path) -> tuple[str | None, str | None]:
    """Повернути (target_subdir_name, kind).

    kind ∈ {"nb", "nb_aggregated", "pointer", "unknown"}
    target_subdir_name = ім'я підпапки (без user_X/) або None якщо не міграбельне.
    """
    stem = pkl_file.stem  # без .pkl
    if not stem.startswith("model_"):
        return None, "unknown"
    exp_part = stem[len("model_"):]  # все після "model_"

    # Spier у bundle, якщо joblib доступний — це найточніше.
    btype, pipe_type = _peek_bundle_type(pkl_file)

    # distilbert/gnn bundle містить "model_dir" поле, type ∈ {distilbert, gnn}
    if btype in ("distilbert", "deberta", "bert", "gnn"):
        return None, "pointer"

    # NB aggregated
    if (btype == "nb" and pipe_type == "aggregated") or exp_part.startswith("nb_aggregated"):
        clean = exp_part.replace("nb_aggregated_", "").replace("nb_aggregated", "")
        clean = clean or "default"
        return f"nb_aggregated_{clean}", "nb_aggregated"

    # NB article-level
    if btype == "nb" or exp_part.startswith("nb_"):
        clean = exp_part[3:] if exp_part.startswith("nb_") else exp_part
        return f"nb_{clean}", "nb"

    # Невідомий .pkl у user_X/ — припускаємо NB (бо це історичний default)
    return f"nb_{exp_part}", "nb"


def migrate_user_dir(user_dir: Path, dry_run: bool) -> tuple[int, int]:
    """Міграція одного user_X каталогу.
    Returns: (n_migrated, n_skipped)
    """
    migrated = 0
    skipped = 0

    for pkl_file in sorted(user_dir.glob("model_*.pkl")):
        if not pkl_file.is_file():
            continue

        target_name, kind = _classify_pkl(pkl_file)

        if kind == "pointer":
            print(f"  ↪ POINTER (distilbert/gnn bundle, skip): {pkl_file.name}")
            skipped += 1
            continue

        if target_name is None:
            print(f"  ⏭ SKIP (unrecognized): {pkl_file.name}")
            skipped += 1
            continue

        target_dir = user_dir / target_name
        target_model = target_dir / "model.pkl"

        if target_model.exists():
            print(f"  ⏭ SKIP (already migrated): {target_model.relative_to(user_dir.parent)}")
            skipped += 1
            continue

        if dry_run:
            print(f"  🔍 WOULD COPY: {pkl_file.name} → {target_dir.name}/model.pkl  ({kind})")
            migrated += 1
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pkl_file, target_model)
        print(f"  ✓ COPIED: {pkl_file.name} → {target_dir.name}/model.pkl  ({kind})")
        migrated += 1

    return migrated, skipped


def cleanup_orphan_predictions(models_root: Path, dry_run: bool) -> int:
    """Видалити predictions.json на рівні user_X/ (orphans)."""
    removed = 0
    for user_dir in sorted(models_root.glob("user_*")):
        if not user_dir.is_dir():
            continue
        orphan = user_dir / "predictions.json"
        if orphan.exists() and orphan.is_file():
            if dry_run:
                print(f"  🔍 WOULD REMOVE orphan: {orphan}")
            else:
                orphan.unlink()
                print(f"  ✓ REMOVED orphan: {orphan}")
            removed += 1
    return removed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Показати що буде зроблено без змін")
    parser.add_argument("--models-root", type=str, default=None,
                        help="Override MODELS_ROOT (за замовчуванням — auto-detect)")
    args = parser.parse_args()

    if args.models_root:
        models_root = Path(args.models_root).expanduser()
        if not models_root.exists():
            print(f"❌ MODELS_ROOT не існує: {models_root}")
            return 1
    else:
        models_root = find_models_root()
        if not models_root:
            print("❌ MODELS_ROOT не знайдено. Спробуй явно через --models-root <path>.")
            print("   Перевірені кандидати:")
            for p in MODELS_ROOT_CANDIDATES:
                print(f"     - {p}")
            return 1

    print(f"📂 MODELS_ROOT: {models_root}")
    print(f"🔧 Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print()

    print("Step 1: Migrating model_*.pkl files до підпапок...")
    total_migrated = 0
    total_skipped = 0
    for user_dir in sorted(models_root.glob("user_*")):
        if not user_dir.is_dir():
            continue
        print(f"\n• {user_dir.name}/")
        m, s = migrate_user_dir(user_dir, dry_run=args.dry_run)
        total_migrated += m
        total_skipped += s

    print("\nStep 2: Cleanup orphan predictions.json...")
    orphans = cleanup_orphan_predictions(models_root, dry_run=args.dry_run)

    print("\n" + "=" * 60)
    print("ПІДСУМОК:")
    verb = "би переміщено" if args.dry_run else "переміщено"
    verb_orphan = "би видалено" if args.dry_run else "видалено"
    print(f"  Моделей {verb}: {total_migrated}")
    print(f"  Пропущено: {total_skipped}")
    print(f"  Orphan predictions.json {verb_orphan}: {orphans}")

    if args.dry_run:
        print(f"\n💡 Запусти без --dry-run щоб виконати: python {sys.argv[0]}")
    else:
        print("\n✅ Migration complete.")
        print("\n⚠️ Оригінальні .pkl файли НЕ видалені (для backward compat — БД ще")
        print("   може посилатись на них). Після того як впевнились що все ОК:")
        print(f"   find {models_root} -maxdepth 2 -name 'model_*.pkl' -type f")

    return 0


if __name__ == "__main__":
    sys.exit(main())
