from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.session import engine
from app.db.base import Base
from app.models import experiment
from app.api import experiments, datasets
from app.api import results
from app.api import reports


# Create database tables
Base.metadata.create_all(bind=engine)

# Create FastAPI app FIRST
app = FastAPI(
    title="SCALE-GNN Test Prioritization System",
    description="AI-based scalable test case prioritization using Graph Neural Networks",
    version="1.0.0"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers AFTER app creation
app.include_router(experiments.router)
app.include_router(datasets.router)
app.include_router(results.router)
app.include_router(reports.router)

# Root endpoint
@app.get("/")
def root():
    return {
        "message": "SCALE-GNN Backend is running",
        "status": "OK"
    }
