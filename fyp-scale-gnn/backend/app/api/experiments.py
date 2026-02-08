from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.experiment import Experiment
from app.models.dataset import Dataset
from app.schemas.experiment import ExperimentResponse
from app.service.runner import run_scale_gnn

router = APIRouter(prefix="/experiments", tags=["Experiments"])


# -------------------- DB Dependency --------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -------------------- RUN EXPERIMENT DIRECTLY FROM DATASET --------------------
# @router.post("/run/{dataset_id}", response_model=ExperimentResponse)
# def run_experiment_direct(
#     dataset_id: int,
#     background_tasks: BackgroundTasks,
#     db: Session = Depends(get_db)
# ):
#     # Check dataset exists
#     dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
#     if not dataset:
#         raise HTTPException(status_code=404, detail="Dataset not found")

#     # Create a new experiment automatically
#     experiment = Experiment(
#         name=f"Experiment for {dataset.name}",
#         dataset_id=dataset.id,
#         parameters={},  # default empty
#         status="PENDING"
#     )
#     db.add(experiment)
#     db.commit()
#     db.refresh(experiment)

#     # Start running in background
#     background_tasks.add_task(run_scale_gnn, experiment.id)

#     # Update dataset status
#     dataset.status = "running"
#     db.commit()

#     return experiment


# -------------------- LIST ALL EXPERIMENTS --------------------
@router.get("/", response_model=list[ExperimentResponse])
def list_experiments(db: Session = Depends(get_db)):
    return (
        db.query(Experiment)
        .order_by(Experiment.created_at.desc())
        .all()
    )


# -------------------- LATEST EXPERIMENTS --------------------
@router.get("/latest", response_model=list[ExperimentResponse])
def latest_experiments(limit: int = 5, db: Session = Depends(get_db)):
    return (
        db.query(Experiment)
        .order_by(Experiment.created_at.desc())
        .limit(limit)
        .all()
    )


# -------------------- PENDING EXPERIMENTS --------------------
@router.get("/pending", response_model=list[ExperimentResponse])
def pending_experiments(db: Session = Depends(get_db)):
    return (
        db.query(Experiment)
        .filter(Experiment.status == "PENDING")
        .order_by(Experiment.created_at.desc())
        .all()
    )


# -------------------- GET SINGLE EXPERIMENT --------------------
@router.get("/{experiment_id}", response_model=ExperimentResponse)
def get_experiment(experiment_id: int, db: Session = Depends(get_db)):
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment


# -------------------- RUN EXISTING EXPERIMENT --------------------
@router.post("/{experiment_id}/run")
def run_existing_experiment(
    experiment_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    experiment = (
        db.query(Experiment)
        .filter(Experiment.id == experiment_id)
        .first()
    )

    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    if experiment.status != "PENDING":
        raise HTTPException(
            status_code=400,
            detail=f"Experiment already {experiment.status}"
        )

    # Update experiment status
    experiment.status = "RUNNING"

    # Update dataset status
    dataset = (
        db.query(Dataset)
        .filter(Dataset.id == experiment.dataset_id)
        .first()
    )
    if dataset:
        dataset.status = "running"

    db.commit()

    # Run in background
    background_tasks.add_task(run_scale_gnn, experiment.id)

    return {
        "message": "Experiment started",
        "experiment_id": experiment.id,
        "status": "RUNNING"
    }

