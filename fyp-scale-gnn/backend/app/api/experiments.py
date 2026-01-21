from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.experiment import Experiment
from app.models.dataset import Dataset
from app.schemas.experiment import ExperimentCreate, ExperimentResponse
from fastapi import BackgroundTasks
from app.service.runner import run_scale_gnn

router = APIRouter(prefix="/experiments", tags=["Experiments"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/", response_model=ExperimentResponse)
def create_experiment(
    data: ExperimentCreate,
    db: Session = Depends(get_db)
):
    dataset = db.query(Dataset).filter(Dataset.id == data.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    experiment = Experiment(
        name=data.name,
        dataset_id=data.dataset_id,
        parameters=data.parameters,
        status="PENDING"
    )

    db.add(experiment)
    db.commit()
    db.refresh(experiment)

    return experiment


@router.get("/", response_model=list[ExperimentResponse])
def list_experiments(db: Session = Depends(get_db)):
    return db.query(Experiment).order_by(Experiment.created_at.desc()).all()


@router.get("/{experiment_id}", response_model=ExperimentResponse)
def get_experiment(experiment_id: int, db: Session = Depends(get_db)):
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment


@router.post("/{experiment_id}/run")
def run_experiment(
    experiment_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    if experiment.status != "PENDING":
        raise HTTPException(status_code=400, detail="Experiment already started")

    background_tasks.add_task(run_scale_gnn, experiment_id)


    return {
        "message": "Experiment started",
        "experiment_id": experiment_id
    }