# scale_gnn/features.py
from __future__ import annotations

from typing import Set, Tuple

import numpy as np
import pandas as pd


def _compute_history_stats(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(
            columns=[
                "test_id",
                "hist_fail_rate",
                "hist_fail_count",
                "duration_p50",
                "duration_std",
                "recency",
                "flakiness",
            ]
        )

    h = history.copy()
    h["is_fail"] = (h["status"] == "fail").astype(int)

    g = h.groupby("test_id")
    max_run = h["run_id"].max()

    def _recency(group: pd.DataFrame) -> float:
        fails = group[group["is_fail"] == 1]
        if fails.empty:
            return 0.0
        return float(fails["run_id"].max() / max_run) if max_run > 0 else 0.0

    def _flakiness(group: pd.DataFrame) -> float:
        s = group.sort_values("run_id")["is_fail"].to_numpy()
        if len(s) < 2:
            return 0.0
        transitions = np.sum(s[1:] != s[:-1])
        return float(transitions / (len(s) - 1))

    stats = pd.DataFrame(
        {
            "test_id": list(g.groups.keys()),
            "hist_fail_rate": g["is_fail"].mean().to_list(),
            "hist_fail_count": g["is_fail"].sum().to_list(),
            "duration_p50": g["time_s"].median().to_list(),
            "duration_std": g["time_s"].std().fillna(0.0).to_list(),
            "recency": [ _recency(group) for _, group in g ],
            "flakiness": [ _flakiness(group) for _, group in g ],
        }
    )

    return stats


def compute_features(
    edges: pd.DataFrame,
    history: pd.DataFrame,
    changed_files: Set[str],
) -> pd.DataFrame:
    """
    Compute per-test features:

      - deg
      - chg_overlap
      - chg_prop
      - hist_fail_rate
      - hist_fail_count
      - duration_p50
      - duration_std
      - recency
      - flakiness
      - changed_flag
    """
    # Structural features from edges
    deg = (
        edges.groupby("test_id")["src_file"]
        .nunique()
        .rename("deg")
        .reset_index()
    )

    chg = (
        edges.assign(changed=edges["src_file"].isin(changed_files))
        .groupby("test_id")["changed"]
        .sum()
        .rename("chg_overlap")
        .reset_index()
    )

    feat = deg.merge(chg, on="test_id", how="left").fillna({"chg_overlap": 0})
    feat["chg_prop"] = feat["chg_overlap"] / feat["deg"].clip(lower=1)

    # History-based features
    hist_stats = _compute_history_stats(history)

    feat = feat.merge(hist_stats, on="test_id", how="left")
    # Fill missing history stats with 0
    for col in [
        "hist_fail_rate",
        "hist_fail_count",
        "duration_p50",
        "duration_std",
        "recency",
        "flakiness",
    ]:
        if col not in feat.columns:
            feat[col] = 0.0
        feat[col] = feat[col].fillna(0.0)

    feat["changed_flag"] = (feat["chg_overlap"] > 0).astype(int)

    return feat
