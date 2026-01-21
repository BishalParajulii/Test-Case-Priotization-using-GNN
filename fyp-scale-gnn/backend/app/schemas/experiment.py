from pydantic import BaseModel
from datetime import datetime
from typing import Dict, Any

class ExperimentCreate(BaseModel):
    name: str
    dataset_id: int
    parameters: Dict[str, Any]

class ExperimentResponse(BaseModel):
    id: int
    name: str
    status: str
    parameters: Dict[str, Any]
    created_at: datetime

    class Config:
        from_attributes = True
