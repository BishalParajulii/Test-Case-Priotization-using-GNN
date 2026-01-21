from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans


@dataclass
class ReductionStats:
    strategy_name: str
    strategy_params: Dict[str, float]

    orig_tests: int
    orig_sources: int
    orig_edges: int

    red_tests: int
    red_sources: int
    red_edges: int

    tests_retention_ratio: float
    edges_retention_ratio: float

    jaccard_fidelity_mean: float
    jaccard_fidelity_std: float


def _compute_point_biserial_per_file(
    edges: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute a point-biserial-like correlation for each src_file between:
      x = coverage of that file (0/1)
      y = failure label (0/1)

    """
    if history.empty:
        # No history, nothing to correlate; everything is zero
        return pd.DataFrame({"src_file": edges["src_file"].unique(), "r_pb": 0.0})

    # global stats
    N = len(history)
    y = (history["status"] == "fail").astype(int)
    total_fail = int(y.sum())
    p_y = total_fail / N
    # std of binary y
    s_y = np.sqrt(p_y * (1.0 - p_y))
    if s_y == 0:
        # all pass or all fail → no signal
        return pd.DataFrame({"src_file": edges["src_file"].unique(), "r_pb": 0.0})

    # Join history with edges so each (run_id, test_id) expands to files it covers
    he = history[["run_id", "test_id", "status"]].merge(
        edges[["test_id", "src_file"]], on="test_id", how="inner"
    )
    he["is_fail"] = (he["status"] == "fail").astype(int)

    # For each file: number of observations with x=1 and mean failure in that group
    g = he.groupby("src_file").agg(
        n1=("test_id", "count"),
        fail1=("is_fail", "sum"),
    ).reset_index()

    g["N"] = N
    g["global_fail"] = total_fail

    # Complement group where x=0 (no coverage of that file)
    g["n0"] = g["N"] - g["n1"]
    g["fail0"] = g["global_fail"] - g["fail1"]

    # Means of y in each subgroup
    g["mu1"] = g["fail1"] / g["n1"].clip(lower=1)
    g["mu0"] = g["fail0"] / g["n0"].clip(lower=1)

    # Proportion of x=1 vs x=0
    g["p"] = g["n1"] / g["N"]
    g["q"] = 1.0 - g["p"]

    # Point-biserial correlation r_pb ~= (mu1 - mu0) / s_y * sqrt(p*q)
    g["r_pb"] = (g["mu1"] - g["mu0"]) / s_y * np.sqrt(g["p"] * g["q"])

    # Some files might not appear in history join at all (no coverage in historical runs)
    # For them, set r_pb = 0.
    all_files = pd.Series(edges["src_file"].unique(), name="src_file")
    g = all_files.to_frame().merge(g[["src_file", "r_pb"]], on="src_file", how="left")
    g["r_pb"] = g["r_pb"].fillna(0.0)

    return g[["src_file", "r_pb"]]


def _compute_jaccard_fidelity(
    edges_orig: pd.DataFrame,
    edges_red: pd.DataFrame,
) -> Tuple[float, float]:
    """
    Because pruning only removes files, the reduced coverage per test is a subset
    of the original coverage.

    For each test present in both:
      J(test) = |C_red(test)| / |C_orig(test)|

    Return mean and std over tests.
    """
    if edges_orig.empty:
        return 1.0, 0.0

    orig_deg = (
        edges_orig.groupby("test_id")["src_file"]
        .nunique()
        .rename("orig_deg")
        .reset_index()
    )
    red_deg = (
        edges_red.groupby("test_id")["src_file"]
        .nunique()
        .rename("red_deg")
        .reset_index()
    )

    merged = orig_deg.merge(red_deg, on="test_id", how="left").fillna({"red_deg": 0})
    merged = merged[merged["orig_deg"] > 0]

    jacc = merged["red_deg"] / merged["orig_deg"]
    if jacc.empty:
        return 1.0, 0.0

    return float(jacc.mean()), float(jacc.std())


def fault_centric_pruning(
    edges: pd.DataFrame,
    history: pd.DataFrame,
    threshold: float = 0.2,
    strategy_name: str = "fault_centric_pruning",
) -> Tuple[pd.DataFrame, ReductionStats, pd.DataFrame]:
    """
    Ω_prune: Remove low-signal files based on file–failure association.

    Steps:
      1) Compute per-file point-biserial-like correlation r_pb.
      2) Keep files where |r_pb| >= threshold.
      3) Drop edges on low-signal files.
      4) Drop tests with no remaining edges.
      5) Compute structural + fidelity stats.

    Returns:
      pruned_edges, stats (dataclass), correlations_df
    """
    edges_orig = edges.copy()

    # Original stats
    orig_tests = edges_orig["test_id"].nunique()
    orig_sources = edges_orig["src_file"].nunique()
    orig_edges = len(edges_orig)

    corr_df = _compute_point_biserial_per_file(edges_orig, history)
    # Decide which files to keep
    keep_mask = corr_df["r_pb"].abs() >= threshold
    kept_files = set(corr_df.loc[keep_mask, "src_file"].tolist())

    if not kept_files:
        # Degenerate case: we pruned everything → just return original with zeros
        kept_files = set(edges_orig["src_file"].unique())

    pruned_edges = edges_orig[edges_orig["src_file"].isin(kept_files)].copy()
    pruned_edges = pruned_edges.drop_duplicates(subset=["test_id", "src_file"])

    # Drop tests that lost all edges
    tests_with_edges = pruned_edges["test_id"].unique().tolist()
    pruned_edges = pruned_edges[pruned_edges["test_id"].isin(tests_with_edges)]

    # Reduced stats
    red_tests = pruned_edges["test_id"].nunique()
    red_sources = pruned_edges["src_file"].nunique()
    red_edges = len(pruned_edges)

    tests_retention_ratio = red_tests / orig_tests if orig_tests > 0 else 1.0
    edges_retention_ratio = red_edges / orig_edges if orig_edges > 0 else 1.0

    j_mean, j_std = _compute_jaccard_fidelity(edges_orig, pruned_edges)

    stats = ReductionStats(
        strategy_name=strategy_name,
        strategy_params={"threshold": threshold},

        orig_tests=orig_tests,
        orig_sources=orig_sources,
        orig_edges=orig_edges,

        red_tests=red_tests,
        red_sources=red_sources,
        red_edges=red_edges,

        tests_retention_ratio=tests_retention_ratio,
        edges_retention_ratio=edges_retention_ratio,

        jaccard_fidelity_mean=j_mean,
        jaccard_fidelity_std=j_std,
    )

    return pruned_edges, stats, corr_df


def append_reduction_summary(
    stats: ReductionStats,
    out_path: Path,
) -> None:
    """
    Append a row to reduction_summary.csv (create if missing).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    row = asdict(stats)
    if out_path.exists():
        df = pd.read_csv(out_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(out_path, index=False)

def spectral_coarsen_tests(
    edges: pd.DataFrame,
    k_clusters: int,
    strategy_name: str = "spectral_coarsen",
    random_state: int = 1337,
) -> Tuple[pd.DataFrame, ReductionStats, pd.DataFrame]:
    """
    Ψ_cluster: Spectral clustering over tests, then contract each cluster to a "super-test".

    Steps:
      1) Build test–test adjacency from shared files (A = B B^T).
      2) Compute top-k eigenvectors of normalized Laplacian.
      3) Run k-means on eigenvectors → k_clusters clusters.
      4) Contract tests in each cluster to a super-test: union of their files.
      5) Compute reduction stats.

    Returns:
      coarsened_edges: DataFrame[test_id, src_file] where test_id = "cluster_<id>"
      stats: ReductionStats
      mapping_df: DataFrame[test_id, cluster_id]
    """
    edges_orig = edges.copy()

    # Original stats
    orig_tests = edges_orig["test_id"].nunique()
    orig_sources = edges_orig["src_file"].nunique()
    orig_edges = len(edges_orig)

    if orig_tests <= k_clusters:
        # nothing to coarsen meaningfully
        # just map each test to its own cluster
        unique_tests = sorted(edges_orig["test_id"].unique())
        mapping_df = pd.DataFrame(
            {
                "test_id": unique_tests,
                "cluster_id": list(range(len(unique_tests))),
            }
        )
        # Make cluster_x ids
        mapping_df["cluster_test_id"] = mapping_df["cluster_id"].apply(
            lambda cid: f"cluster_{cid}"
        )

        # Join edges to mapping and rename test_id to cluster_test_id
        e = edges_orig.merge(mapping_df[["test_id", "cluster_test_id"]], on="test_id", how="left")
        e = e[["cluster_test_id", "src_file"]].rename(columns={"cluster_test_id": "test_id"})
        e = e.drop_duplicates()

        red_tests = e["test_id"].nunique()
        red_sources = e["src_file"].nunique()
        red_edges = len(e)

        stats = ReductionStats(
            strategy_name=strategy_name,
            strategy_params={"k_clusters": k_clusters, "note": "no-op (tests <= k_clusters)"},
            orig_tests=orig_tests,
            orig_sources=orig_sources,
            orig_edges=orig_edges,
            red_tests=red_tests,
            red_sources=red_sources,
            red_edges=red_edges,
            tests_retention_ratio=1.0,
            edges_retention_ratio=1.0,
            # Coarsening merges tests, so our previous per-test Jaccard definition
            # doesn't apply directly; treat it as structurally faithful=1.0.
            jaccard_fidelity_mean=1.0,
            jaccard_fidelity_std=0.0,
        )
        return e, stats, mapping_df

    # Index tests
    test_ids = sorted(edges_orig["test_id"].unique())
    test2idx = {t: i for i, t in enumerate(test_ids)}
    n_tests = len(test_ids)

    # Index files
    file_ids = sorted(edges_orig["src_file"].unique())
    file2idx = {f: i for i, f in enumerate(file_ids)}
    n_files = len(file_ids)

    # Build sparse incidence matrix B (tests x files)
    row = [test2idx[t] for t in edges_orig["test_id"]]
    col = [file2idx[f] for f in edges_orig["src_file"]]
    data = np.ones(len(row), dtype=np.float32)
    B = sp.coo_matrix((data, (row, col)), shape=(n_tests, n_files)).tocsr()

    # Test–test adjacency: A = B B^T
    A = B @ B.T
    # Remove self loops for spectral clustering
    A.setdiag(0)
    A.eliminate_zeros()

    # Normalized Laplacian
    L = sp.csgraph.laplacian(A, normed=True)

    # Number of clusters should be < n_tests
    k = min(k_clusters, n_tests - 1)
    # Compute k smallest magnitude eigenvectors of L
    # (skip the trivial one if you want, but for now keep them)
    eigvals, eigvecs = eigsh(L, k=k, which="SM")

    # Rows of eigvecs are node embeddings for clustering
    X_spec = eigvecs

    # K-means on spectral embeddings
    km = KMeans(n_clusters=k_clusters, random_state=random_state, n_init="auto")
    cluster_labels = km.fit_predict(X_spec)

    mapping_df = pd.DataFrame(
        {
            "test_id": test_ids,
            "cluster_id": cluster_labels,
        }
    )
    mapping_df["cluster_test_id"] = mapping_df["cluster_id"].apply(
        lambda cid: f"cluster_{cid}"
    )

    # Contract edges: join original edges with cluster mapping
    e = edges_orig.merge(mapping_df[["test_id", "cluster_test_id"]], on="test_id", how="left")
    e = e[["cluster_test_id", "src_file"]].rename(columns={"cluster_test_id": "test_id"})
    e = e.drop_duplicates().reset_index(drop=True)

    # Reduced stats
    red_tests = e["test_id"].nunique()
    red_sources = e["src_file"].nunique()
    red_edges = len(e)

    tests_retention_ratio = red_tests / orig_tests if orig_tests > 0 else 1.0
    edges_retention_ratio = red_edges / orig_edges if orig_edges > 0 else 1.0

    # Per-test Jaccard doesn't really make sense after merging tests into clusters,
    # but for now we treat coarsening as structure-preserving at file-level.
    j_mean = 1.0
    j_std = 0.0

    stats = ReductionStats(
        strategy_name=strategy_name,
        strategy_params={"k_clusters": k_clusters},
        orig_tests=orig_tests,
        orig_sources=orig_sources,
        orig_edges=orig_edges,
        red_tests=red_tests,
        red_sources=red_sources,
        red_edges=red_edges,
        tests_retention_ratio=tests_retention_ratio,
        edges_retention_ratio=edges_retention_ratio,
        jaccard_fidelity_mean=j_mean,
        jaccard_fidelity_std=j_std,
    )

    return e, stats, mapping_df


def save_cluster_mapping(mapping_df: pd.DataFrame, out_path: Path) -> None:
    """
    Save mapping from original tests to spectral clusters as CSV.
    Columns: test_id, cluster_id, cluster_test_id
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(out_path, index=False)
