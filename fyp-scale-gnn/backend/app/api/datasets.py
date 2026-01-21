import os
import shutil
from fastapi import APIRouter, UploadFile, File, Form, Depends
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.dataset import Dataset
from app.schemas.dataset import DatasetResponse
from app.core.config import settings

router = APIRouter(prefix="/datasets", tags=["Datasets"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/upload", response_model=DatasetResponse)
def upload_dataset(
    name: str = Form(...),
    coverage_edges: UploadFile = File(...),
    history: UploadFile = File(...),
    test_runs: UploadFile = File(...),
    changed_files: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    dataset = Dataset(name=name, path="")
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    dataset_dir = os.path.join(settings.DATA_DIR, f"dataset_{dataset.id}")
    os.makedirs(dataset_dir, exist_ok=True)

    files = {
        "coverage_edges.csv": coverage_edges,
        "history.csv": history,
        "test_runs.csv": test_runs,
        "changed_files.txt": changed_files,
    }

    for filename, file in files.items():
        with open(os.path.join(dataset_dir, filename), "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    dataset.path = dataset_dir
    db.commit()

    return dataset
