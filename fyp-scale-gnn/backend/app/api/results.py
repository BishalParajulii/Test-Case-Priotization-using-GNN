from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.experiment import Experiment
from app.service.result_service import (
    load_ranking,
    load_top_tests,
    load_metrics,
    load_reduction_stats,
)

router = APIRouter(prefix="/results", tags=["Results"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _validate_experiment(experiment_id: int, db: Session) -> Experiment:
    experiment = (
        db.query(Experiment)
        .filter(Experiment.id == experiment_id)
        .first()
    )

    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    if experiment.status != "COMPLETED":
        raise HTTPException(
            status_code=400,
            detail=f"Experiment status is {experiment.status}"
        )

    return experiment


# ============================
# Combined results (charts)
# ============================

@router.get("/{experiment_id}")
def get_full_results(experiment_id: int, db: Session = Depends(get_db)):
    _validate_experiment(experiment_id, db)

    return {
        "experiment_id": experiment_id,
        "ranking": load_ranking(experiment_id),
        "reduction": load_reduction_stats(experiment_id),
        "metrics": load_metrics(experiment_id),
        "top_tests": load_top_tests(experiment_id),
    }


# ============================
# Individual endpoints
# ============================

@router.get("/{experiment_id}/ranking")
def get_ranking_api(experiment_id: int, db: Session = Depends(get_db)):
    _validate_experiment(experiment_id, db)
    return load_ranking(experiment_id)


@router.get("/{experiment_id}/top-tests")
def get_top_tests_api(experiment_id: int, db: Session = Depends(get_db)):
    _validate_experiment(experiment_id, db)
    return load_top_tests(experiment_id)


@router.get("/{experiment_id}/metrics")
def get_metrics_api(experiment_id: int, db: Session = Depends(get_db)):
    _validate_experiment(experiment_id, db)
    return load_metrics(experiment_id)


@router.get("/{experiment_id}/reduction")
def get_reduction_api(experiment_id: int, db: Session = Depends(get_db)):
    _validate_experiment(experiment_id, db)
    return load_reduction_stats(experiment_id)
