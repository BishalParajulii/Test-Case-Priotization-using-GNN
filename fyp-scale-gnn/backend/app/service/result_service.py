import os
import json
import pandas as pd
from typing import List, Dict

from app.core.config import settings


def _experiment_dir(experiment_id: int) -> str:
    return os.path.join(settings.OUTPUT_DIR, f"experiment_{experiment_id}")


def _rank_dir(experiment_id: int) -> str:
    return os.path.join(_experiment_dir(experiment_id), "rank")


# ============================
# Ranking
# ============================

def load_ranking(experiment_id: int) -> List[Dict]:
    path = os.path.join(_rank_dir(experiment_id), "ranks_scale_gcn.csv")
    if not os.path.exists(path):
        return []
    return pd.read_csv(path).to_dict(orient="records")


def load_top_tests(experiment_id: int) -> List[str]:
    path = os.path.join(_rank_dir(experiment_id), "top_k_tests.txt")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [line.strip() for line in f.readlines()]


def load_metrics(experiment_id: int) -> Dict:
    path = os.path.join(_rank_dir(experiment_id), "metrics_runtime.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# ============================
# Reduction statistics
# ============================

def load_reduction_stats(experiment_id: int) -> Dict:
    """
    Loads reduction_summary.csv from fault_prune & spectral stages
    """
    base = _experiment_dir(experiment_id)
    stats = {}

    for stage in ["fault_prune", "spectral"]:
        path = os.path.join(base, stage, "reduction_summary.csv")
        if os.path.exists(path):
            stats[stage] = pd.read_csv(path).to_dict(orient="records")

    return stats
