import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from api.database import SessionLocal, ModelRecord


def main():
    db = SessionLocal()
    models = db.query(ModelRecord).filter(
        ModelRecord.predictions_json.isnot(None),
        ModelRecord.predictions_json != "",
    ).all()
    print(f"Found {len(models)} models with predictions_json\n")
    fixed = 0
    for m in models:
        try:
            data = json.loads(m.predictions_json)
        except Exception as e:
            print(f"  WARN id={m.id} '{m.name}': JSON error: {e}")
            continue
        current = data.get("splits_used")
        target = m.splits_used
        if current == target:
            continue
        if target:
            print(f"  FIX id={m.id} [{m.model_type}] '{current}' -> '{target}'")
            data["splits_used"] = target
            m.predictions_json = json.dumps(data)
            fixed += 1
    if fixed > 0:
        db.commit()
        print(f"\nFixed: {fixed} models")
    else:
        print(f"\nAll already OK")
    print("\n" + "=" * 70)
    print("VERIFICATION:")
    print("=" * 70)
    for m in db.query(ModelRecord).filter(
        ModelRecord.predictions_json.isnot(None)
    ).order_by(ModelRecord.id.desc()).all():
        try:
            preds_split = json.loads(m.predictions_json).get("splits_used", "?")
        except Exception:
            preds_split = "PARSE_ERROR"
        ok = "OK" if preds_split == m.splits_used else "FAIL"
        print(f"  [{ok}] id={m.id} [{m.model_type}] db={m.splits_used} preds={preds_split}")
    db.close()


if __name__ == "__main__":
    main()