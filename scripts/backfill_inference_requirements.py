"""One-time backfill: populate `ModelRecord.inference_requirements` для
legacy записів, які тренувались до того, як ми ввели цей column.

Запуск: `python -m scripts.backfill_inference_requirements [--dry-run]`

Idempotent: пропускає записи де requirements уже виставлені (non-empty
JSON). --force перезаписує всім за дефолтами з derive_requirements.
"""
from __future__ import annotations

import argparse
import json
import sys

from api.database import ModelRecord, SessionLocal, create_tables
from api.inference_context import derive_requirements


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Лише друкувати, без commit")
    parser.add_argument("--force", action="store_true",
                        help="Перезаписати existing requirements")
    args = parser.parse_args(argv)

    create_tables()  # переконатися що колонка інстальована
    session = SessionLocal()
    try:
        records = session.query(ModelRecord).all()
        updated = 0
        skipped = 0
        for r in records:
            existing = r.inference_requirements
            if existing and not args.force:
                skipped += 1
                continue
            reqs = derive_requirements(
                model_type=r.model_type or "",
                pipeline_type=r.pipeline_type,
            )
            print(
                f"  id={r.id:3} type={r.model_type:12} pipeline={r.pipeline_type:12} "
                f"→ claim={reqs['claim_extraction']} "
                f"social={(reqs['social_search'] or {}).get('enabled', False)} "
                f"graph={(reqs['graph_construction'] or {}).get('enabled', False)}"
            )
            if not args.dry_run:
                r.inference_requirements = json.dumps(reqs, ensure_ascii=False)
                updated += 1

        if args.dry_run:
            print(f"\n[dry-run] would update {len(records) - skipped} records, skip {skipped}")
        else:
            session.commit()
            print(f"\nupdated {updated}, skipped {skipped} (already set)")
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())
