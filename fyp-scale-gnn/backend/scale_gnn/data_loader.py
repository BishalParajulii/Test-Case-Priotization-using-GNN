# scale_gnn/data_loader.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Set

import pandas as pd


def load_edges(path: Path) -> pd.DataFrame:
    """
    Load coverage edges: test_id, src_file.
    Cleans nulls, strips whitespace, and drops duplicates.
    """
    if not path.exists():
        raise FileNotFoundError(f"edges file not found: {path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    required = {"test_id", "src_file"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must have columns {required}, found {set(df.columns)}")

    df["test_id"] = df["test_id"].astype(str).str.strip()
    df["src_file"] = df["src_file"].astype(str).str.strip()

    df = df.dropna(subset=["test_id", "src_file"])
    df = df[(df["test_id"] != "") & (df["src_file"] != "")]
    df = df.drop_duplicates(subset=["test_id", "src_file"]).reset_index(drop=True)
    return df


def _load_results_generic(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"results file not found: {path}")

    df = pd.read_csv(path)
    required = {"run_id", "test_id", "status"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must have columns {required}, found {set(df.columns)}")

    df["run_id"] = pd.to_numeric(df["run_id"], errors="coerce")
    df["test_id"] = df["test_id"].astype(str).str.strip()
    df["status"] = df["status"].astype(str).str.lower().str.strip()

    if "time_s" not in df.columns:
        df["time_s"] = 0.0
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce").fillna(0.0)

    df = df[df["status"].isin(["pass", "fail"])].copy()
    df = df.dropna(subset=["run_id", "test_id"])
    df = df.reset_index(drop=True)
    return df


def load_history(path: Path) -> pd.DataFrame:
    """
    Load historical test results (multiple runs).
    """
    return _load_results_generic(path)


def load_current(path: Path) -> pd.DataFrame:
    """
    Load current test run results used for evaluation.
    """
    return _load_results_generic(path)


def load_changed_files(path: Path) -> Set[str]:
    """
    Load changed files list as a set of normalized paths.
    If file doesn't exist, returns empty set.
    """
    if not path.exists():
        return set()

    lines = path.read_text(encoding="utf-8").splitlines()
    files = {ln.strip() for ln in lines if ln.strip()}
    return files


def load_all(
    edges_path: Path,
    history_path: Path,
    current_path: Path,
    changed_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Set[str]]:
    """
    Convenience loader for the full dataset.
    """
    edges = load_edges(edges_path)
    history = load_history(history_path)
    current = load_current(current_path)
    changed = load_changed_files(changed_path)
    return edges, history, current, changed
