# scale_gnn/config.py

from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataPaths:
    edges: Path          # coverage_edges.csv
    history: Path        # history.csv
    current: Path        # test_runs.csv
    changed: Path        # changed_files.txt

@dataclass
class ScaleGNNConfig:
    data: DataPaths
    out_dir: Path
    seed: int = 1337
