from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    PROJECT_NAME = "SCALE-GNN"
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./scale_gnn.db")
    DATA_DIR = "data"
    OUTPUT_DIR = "out"
    REPORT_DIR = "reports"

settings = Settings()
