import os, glob
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db, ModelRecord, User
from app.auth import get_current_user
from app.schemas import ModelRecordResponse

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=List[ModelRecordResponse])
def list_models(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    return db.query(ModelRecord).order_by(ModelRecord.created_at.desc()).all()


@router.patch("/{model_id}/activate", response_model=ModelRecordResponse)
def activate_model(
    model_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    target = db.query(ModelRecord).filter(ModelRecord.id == model_id).first()
    if not target:
        raise HTTPException(status_code=404, detail="Model not found")

    base_path = os.path.join(os.path.dirname(__file__), "../..")
    pkl_path = os.path.join(base_path, "models", target.filename)
    if not os.path.exists(pkl_path):
        raise HTTPException(status_code=404, detail="Model .pkl file missing from disk")

    # Deactivate all, then activate target
    db.query(ModelRecord).update({"is_active": False})
    target.is_active = True
    db.commit()
    db.refresh(target)

    # Hot-swap the live detector instance
    from app.main import detector
    detector.load_from_file(pkl_path)

    return target


@router.delete("", status_code=200)
def delete_all_models(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    models = db.query(ModelRecord).all()
    if not models:
        raise HTTPException(status_code=404, detail="No models to delete")

    base_path = os.path.join(os.path.dirname(__file__), "../..")
    models_dir = os.path.join(base_path, "models")

    # Delete .pkl files from disk
    for m in models:
        pkl_path = os.path.join(models_dir, m.filename)
        if os.path.exists(pkl_path):
            os.remove(pkl_path)

    count = len(models)
    db.query(ModelRecord).delete()
    db.commit()

    return {"message": f"Видалено {count} моделей"}
