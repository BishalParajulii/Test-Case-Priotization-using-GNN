import os
import shutil
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.dataset import Dataset
from app.schemas.dataset import DatasetResponse
from app.core.config import settings

router = APIRouter(prefix="/datasets", tags=["Datasets"])


# -------------------- DB Dependency --------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -------------------- UPLOAD DATASET --------------------
@router.post("/upload", response_model=DatasetResponse)
def upload_dataset(
    name: str = Form(...),
    coverage_edges: UploadFile = File(...),
    history: UploadFile = File(...),
    test_runs: UploadFile = File(...),
    changed_files: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
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
            file_path = os.path.join(dataset_dir, filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        dataset.path = dataset_dir
        db.commit()
        return dataset

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


# -------------------- LATEST DATASET --------------------
@router.get("/latest", response_model=DatasetResponse)
def get_latest_dataset(db: Session = Depends(get_db)):
    dataset = (
        db.query(Dataset)
        .order_by(Dataset.id.desc())
        .first()
    )

    if not dataset:
        raise HTTPException(status_code=404, detail="No dataset found")

    return dataset


# -------------------- LIST ALL DATASETS --------------------
@router.get("/", response_model=list[DatasetResponse])
def list_datasets(db: Session = Depends(get_db)):
    return (
        db.query(Dataset)
        .order_by(Dataset.created_at.desc())
        .all()
    )


# -------------------- GET DATASET BY ID --------------------
@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


#---------------------Status ------------------------------

@router.get("/status/{dataset_id}")
def dataset_status(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return {
        "dataset_id": dataset.id,
        "name": dataset.name,
        "status": dataset.status,
        "created_at": dataset.created_at
    }