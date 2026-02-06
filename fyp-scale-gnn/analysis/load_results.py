import pandas as pd
import json
import os
from analysis.config import BASE_EXPERIMENT_DIR

# Load ranking CSV
def load_ranking():
    return pd.read_csv(
        os.path.join(BASE_EXPERIMENT_DIR, "rank/ranks_scale_gcn.csv")
    )

# Load metrics JSON
def load_metrics():
    with open(
        os.path.join(BASE_EXPERIMENT_DIR, "rank/metrics_runtime.json")
    ) as f:
        return json.load(f)

# Load reduction CSVs
def load_reduction():
    fault = pd.read_csv(
        os.path.join(BASE_EXPERIMENT_DIR, "fault_prune/reduction_summary.csv")
    )
    spectral = pd.read_csv(
        os.path.join(BASE_EXPERIMENT_DIR, "spectral/reduction_summary.csv")
    )
    return fault, spectral

# Wrapper function to load everything at once
def load_results():
    ranking = load_ranking()
    metrics = load_metrics()
    fault, spectral = load_reduction()
    return ranking, metrics, fault, spectral
