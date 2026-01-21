from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.experiment import Experiment
from app.service.report_service import generate_experiment_report

router = APIRouter(prefix="/reports", tags=["Reports"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/{experiment_id}")
def download_report(experiment_id: int, db: Session = Depends(get_db)):
    experiment = db.query(Experiment).filter(
        Experiment.id == experiment_id
    ).first()

    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    if experiment.status != "COMPLETED":
        raise HTTPException(status_code=400, detail="Experiment not completed")

    path = generate_experiment_report(experiment_id)

    return FileResponse(
        path,
        media_type="application/pdf",
        filename=f"experiment_{experiment_id}_report.pdf"
    )
