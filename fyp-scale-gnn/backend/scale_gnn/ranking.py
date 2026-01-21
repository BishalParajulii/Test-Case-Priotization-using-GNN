# scale_gnn/ranking.py
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from .features import compute_features
from .model import ScaleGCN, train_scale_gcn, predict_scale_gcn


def _build_test_graph_from_edges(edges: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Build test–test adjacency from test–file edges via co-coverage.
    Returns:
      edge_index (2,E), and ordered list of test_ids
    """
    test_ids = sorted(edges["test_id"].unique())
    test2idx = {t: i for i, t in enumerate(test_ids)}
    file_ids = sorted(edges["src_file"].unique())
    file2idx = {f: i for i, f in enumerate(file_ids)}

    row = [test2idx[t] for t in edges["test_id"]]
    col = [file2idx[f] for f in edges["src_file"]]
    data = np.ones(len(row), dtype=np.float32)

    B = sp.coo_matrix((data, (row, col)), shape=(len(test_ids), len(file_ids))).tocsr()
    A = B @ B.T
    A.setdiag(0)
    A.eliminate_zeros()

    coo = A.tocoo()
    edge_index = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    return edge_index, torch.arange(len(test_ids), dtype=torch.long), test_ids


def _normalize_features(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        col = df[c].astype(float)
        mn = col.min()
        mx = col.max()
        if mx > mn:
            df[c] = (col - mn) / (mx - mn)
        else:
            df[c] = 0.0
    return df


def _compute_apfd(order: List[str], failing: List[str]) -> float:
    failing_set = set(failing)
    if not failing_set:
        return 1.0
    n = len(order)
    hits = [order.index(t) + 1 for t in failing if t in order]
    if not hits:
        return 0.0
    return 1.0 - (sum(hits) / (len(failing) * n)) + (1.0 / (2 * n))


def _recall_precision_at_k(order: List[str], failing: List[str], k: int) -> Tuple[float, float]:
    failing_set = set(failing)
    if not failing_set:
        return 1.0, 1.0
    top = order[:k]
    hits = sum(1 for t in top if t in failing_set)
    recall = hits / len(failing_set)
    precision = hits / max(len(top), 1)
    return recall, precision


def rank_tests_with_gnn(
    edges: pd.DataFrame,
    history: pd.DataFrame,
    current: pd.DataFrame,
    changed_files: set[str],
    embeddings_dir: Path,
    cluster_mapping_path: Path,
    out_dir: Path,
    top_k: int = 100,
    num_epochs: int = 100,
) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings + mapping from Θ_encode
    emb_path = embeddings_dir / "embeddings.npy"
    map_path = embeddings_dir / "test_id_to_idx.csv"
    embeddings = np.load(emb_path)
    map_df = pd.read_csv(map_path)

    # Feature engineering
    feat_df = compute_features(edges, history, changed_files)
    # Merge to restrict to tests that have embeddings
    df = map_df.merge(feat_df, on="test_id", how="inner")

    # Align rows by idx for embeddings
    df = df.sort_values("idx").reset_index(drop=True)
    test_ids = df["test_id"].tolist()

    # Numerical feature columns (exclude identifiers)
    feature_cols = [
        "deg",
        "chg_overlap",
        "chg_prop",
        "hist_fail_rate",
        "hist_fail_count",
        "duration_p50",
        "duration_std",
        "recency",
        "flakiness",
        "changed_flag",
    ]
    df = _normalize_features(df, [c for c in feature_cols if c != "changed_flag"])

    feat_mat = df[feature_cols].to_numpy(dtype=np.float32)
    emb_mat = embeddings[df["idx"].to_numpy()]

    x_all = np.concatenate([feat_mat, emb_mat], axis=1)

    # Build test graph
    # Use only edges for tests that have embeddings
    edges_sub = edges[edges["test_id"].isin(test_ids)].copy()
    edge_index, node_idx, _ = _build_test_graph_from_edges(edges_sub)

    x = torch.from_numpy(x_all)

    # Labels: has failed historically?
    hist_fail = (
        history.assign(is_fail=(history["status"] == "fail").astype(int))
        .groupby("test_id")["is_fail"]
        .max()
        .reindex(test_ids)
        .fillna(0)
        .astype(int)
        .to_numpy()
    )

    y = torch.from_numpy(hist_fail)

    # Train GCN
    in_dim = x.shape[1]
    model = ScaleGCN(in_dim=in_dim, hidden_dim=128, dropout=0.2)

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_scale_gcn(
        model,
        x,
        edge_index,
        y,
        num_epochs=num_epochs,
        lr=1e-3,
        weight_decay=1e-4,
        device=device,
    )
    train_time_s = time.time() - start_time

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        gpu_mem_mb = 0.0

    # Predict risk at cluster-level (each node here is a spectral "super-test")
    probs = predict_scale_gcn(model, x, edge_index, device=device).numpy()
    risk_scores = probs

    cluster_ranking = pd.DataFrame(
        {
            "cluster_test_id": test_ids,   # these are the test_ids in coarsened graph (cluster_xxx)
            "risk_score": risk_scores,
        }
    ).sort_values("risk_score", ascending=False)
    cluster_ranking["cluster_rank"] = np.arange(1, len(cluster_ranking) + 1)

    # Expand back to original tests using spectral mapping
    mapping_df = pd.read_csv(cluster_mapping_path)
    # expected columns: test_id (original), cluster_id, cluster_test_id

    expanded = mapping_df.merge(cluster_ranking, on="cluster_test_id", how="inner")
    # Now each original test gets its cluster's risk_score
    expanded = expanded.sort_values("risk_score", ascending=False).reset_index(drop=True)
    expanded["rank"] = np.arange(1, len(expanded) + 1)

    # This is the ranking in original test space
    ranking_df = expanded[["test_id", "cluster_test_id", "risk_score", "rank"]]


    # Metrics vs current run failures in ORIGINAL test space
    failing_current = current[current["status"] == "fail"]["test_id"].unique().tolist()
    order = ranking_df["test_id"].tolist()

    apfd = _compute_apfd(order, failing_current)
    recall_k, prec_k = _recall_precision_at_k(order, failing_current, top_k)

    # Baseline order: use execution order from current.csv (row order)
    baseline_order = current["test_id"].tolist()
    apfd_baseline = _compute_apfd(baseline_order, failing_current)
    delta_apfd = apfd - apfd_baseline

    # Fault retention: which failing tests even appear in our reduced/coarsened/clustered universe
    reduced_tests = set(order)
    failing_set = set(failing_current)
    if failing_set:
        retained_faults = len(failing_set & reduced_tests)
        fault_retention_ratio = retained_faults / len(failing_set)
    else:
        fault_retention_ratio = 1.0


    # Write ranking CSV
    ranks_path = out_dir / "ranks_scale_gcn.csv"
    ranking_df.to_csv(ranks_path, index=False)

    # Write top_k_tests.txt
    top_ids = ranking_df["test_id"].head(top_k).tolist()
    topk_path = out_dir / "top_k_tests.txt"
    topk_path.write_text("\n".join(top_ids), encoding="utf-8")

    # Metrics JSON
    metrics = {
        "apfd": apfd,
        "apfd_baseline": apfd_baseline,
        "delta_apfd": delta_apfd,
        "fault_retention_ratio": fault_retention_ratio,
        "recall_at_k": recall_k,
        "precision_at_k": prec_k,
        "k": top_k,
        "num_tests_ranked": int(len(order)),
        "num_failures_current": int(len(failing_current)),
        "train_time_s": float(train_time_s),
        "gpu_mem_mb": float(gpu_mem_mb),
        "peak_nodes": int(len(order)),
        "peak_edges": int(edge_index.size(1)),
    }


    metrics_path = out_dir / "metrics_runtime.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"[rank] wrote ranking to {ranks_path}")
    print(f"[rank] wrote top-{top_k} tests to {topk_path}")
    print(f"[rank] wrote metrics to {metrics_path}")
