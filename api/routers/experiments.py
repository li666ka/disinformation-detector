from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from api.database import get_db, Experiment, User
from api.auth import get_current_user
from api.schemas import ExperimentResponse

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.get("", response_model=List[ExperimentResponse])
def list_experiments(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return (
        db.query(Experiment)
        .filter(Experiment.user_id == current_user.id)
        .order_by(Experiment.created_at.desc())
        .all()
    )


@router.delete("/{experiment_id}")
def delete_experiment(
    experiment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    exp = db.query(Experiment).filter(
        Experiment.id == experiment_id,
        Experiment.user_id == current_user.id,
    ).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Експеримент не знайдено")
    db.delete(exp)
    db.commit()
    return {"ok": True}


@router.delete("")
def delete_all_experiments(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    count = db.query(Experiment).filter(
        Experiment.user_id == current_user.id,
    ).delete()
    db.commit()
    return {"ok": True, "deleted": count}
