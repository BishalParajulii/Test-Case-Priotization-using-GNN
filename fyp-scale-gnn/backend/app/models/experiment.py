from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db.base import Base
from sqlalchemy.sql import func

class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)

    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    status = Column(String, default="PENDING")

    parameters = Column(JSON)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    dataset = relationship("Dataset")
