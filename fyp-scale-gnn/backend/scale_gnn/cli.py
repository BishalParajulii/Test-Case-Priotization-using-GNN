from __future__ import annotations
import pandas as pd
from rich.console import Console

from .compression import (
    neural_compress_coverage,
    save_embeddings_and_stats,
)
from .ranking import rank_tests_with_gnn

import argparse
from pathlib import Path
import pandas as pd

from .data_loader import load_all
from .reduction import fault_centric_pruning, append_reduction_summary, spectral_coarsen_tests, save_cluster_mapping
from .compression import     neural_compress_coverage,    save_embeddings_and_stats

def cmd_rank(args: argparse.Namespace) -> None:
    console = Console()
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    console.log("[rank] loading data...")

    from .data_loader import load_all
    edges, history, current, changed = load_all(
        args.edges, args.history, args.current, args.changed
    )

    console.log(
        f"[rank] edges={len(edges)}, tests={edges['test_id'].nunique()}, "
        f"files={edges['src_file'].nunique()}"
    )

    rank_tests_with_gnn(
        edges=edges,
        history=history,
        current=current,
        changed_files=changed,
        embeddings_dir=args.embeddings,
        cluster_mapping_path=args.cluster_mapping,
        out_dir=out_dir,
        top_k=args.topk,
        num_epochs=args.epochs,
    )

def cmd_compress(args: argparse.Namespace) -> None:
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    edges_path: Path = args.edges
    if edges_path.suffix.lower() == ".parquet":
        edges = pd.read_parquet(edges_path)
    else:
        edges = pd.read_csv(edges_path)

    console = getattr(args, "console", None)

    if console is None:
        from rich.console import Console
        console = Console()

    console.log(
        f"[Θ_encode] compressing coverage for {edges['test_id'].nunique()} tests "
        f"and {edges['src_file'].nunique()} files..."
    )

    embeddings, test2idx, stats = neural_compress_coverage(
        edges,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        smooth_lambda=args.smooth_lambda,
        epochs=args.epochs,
        lr=args.lr,
    )

    save_embeddings_and_stats(embeddings, test2idx, stats, out_dir)

    console.log(
        f"[Θ_encode] dim reduction: input_dim={stats.input_dim}, "
        f"latent_dim={stats.latent_dim}, "
        f"ratio={stats.reduction_ratio:.4f}"
    )

def cmd_spectral_coarsen(args: argparse.Namespace) -> None:
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    edges_path: Path = args.edges

    if edges_path.suffix.lower() == ".parquet":
        edges = pd.read_parquet(edges_path)
    else:
        edges = pd.read_csv(edges_path)

    print(f"[Ψ_cluster] running spectral coarsening on {len(edges)} edges...")

    coarsened_edges, stats, mapping_df = spectral_coarsen_tests(
        edges,
        k_clusters=args.k_clusters,
        strategy_name="spectral_coarsen",
        random_state=1337,
    )

    # Save gaarxa coarsened edges
    out_edges = out_dir / "edges_spectral_coarsened.parquet"
    coarsened_edges.to_parquet(out_edges, index=False)
    print(f"[Ψ_cluster] wrote coarsened edges to {out_edges}")

    # Save gaarxa mapping
    map_path = out_dir / "spectral_cluster_mapping.csv"
    save_cluster_mapping(mapping_df, map_path)
    print(f"[Ψ_cluster] wrote test→cluster mapping to {map_path}")

    # reduction ko summary append garna ko laagi
    summary_path = out_dir / "reduction_summary.csv"
    append_reduction_summary(stats, summary_path)
    print(f"[Ψ_cluster] updated reduction summary at {summary_path}")

    # Summary print garna ko laagi 
    print("\n[Ψ_cluster] reduction stats:")
    print(
        f"  tests:  {stats.orig_tests} → {stats.red_tests} "
        f"({stats.tests_retention_ratio:.3f} retained)"
    )
    print(
        f"  files:  {stats.orig_sources} → {stats.red_sources}"
    )
    print(
        f"  edges:  {stats.orig_edges} → {stats.red_edges} "
        f"({stats.edges_retention_ratio:.3f} retained)"
    )

def cmd_inspect_data(args: argparse.Namespace) -> None:
    edges, history, current, changed = load_all(
        args.edges, args.history, args.current, args.changed
    )
    print(
        f"edges:   {len(edges)} rows, "
        f"{edges['test_id'].nunique()} tests, "
        f"{edges['src_file'].nunique()} files"
    )
    print(f"history: {len(history)} rows, {history['run_id'].nunique()} runs")
    print(f"current: {len(current)} rows, {current['run_id'].nunique()} runs")
    print(f"changed: {len(changed)} files")


def cmd_reduce(args: argparse.Namespace) -> None:
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load only edges + history
    edges, history, _, _ = load_all(
        args.edges, args.history, args.current, args.changed
    )

    print("[Ω_prune] running fault-centric pruning...")
    pruned_edges, stats, corr_df = fault_centric_pruning(
        edges, history, threshold=args.threshold
    )

    #  pruned edges haru save garna ko laagi
    pruned_path = out_dir / "edges_fault_pruned.parquet"
    pruned_edges.to_parquet(pruned_path, index=False)
    print(f"[Ω_prune] wrote pruned edges to {pruned_path}")

    # inspection ko laagi file-fault correlations save garna ko laagi
    corr_path = out_dir / "file_fault_correlations.csv"
    corr_df.to_csv(corr_path, index=False)
    print(f"[Ω_prune] wrote file–fault correlations to {corr_path}")

    # reduction ko summary append garna ko laagi
    summary_path = out_dir / "reduction_summary.csv"
    append_reduction_summary(stats, summary_path)
    print(f"[Ω_prune] updated reduction summary at {summary_path}")

    # Summary print garna ko laagi
    print("\n[Ω_prune] reduction stats:")
    print(f"  strategy: {stats.strategy_name} (threshold={args.threshold})")
    print(
        f"  tests:  {stats.orig_tests} → {stats.red_tests} "
        f"({stats.tests_retention_ratio:.3f} retained)"
    )
    print(
        f"  files:  {stats.orig_sources} → {stats.red_sources}"
    )
    print(
        f"  edges:  {stats.orig_edges} → {stats.red_edges} "
        f"({stats.edges_retention_ratio:.3f} retained)"
    )
    print(
        f"  Jaccard fidelity (per test): "
        f"mean={stats.jaccard_fidelity_mean:.3f}, "
        f"std={stats.jaccard_fidelity_std:.3f}"
    )


def main():
    parser = argparse.ArgumentParser(description="SCALE-GNN CLI")
    sub = parser.add_subparsers(dest="command", required=True)
# Rank garna ko laagi 
    rank_p = sub.add_parser("rank", help="Rank tests using ScaleGCN on compressed+coarsened graph")
    rank_p.add_argument("--edges", required=True, type=Path, help="Coarsened edges (spectral output)")
    rank_p.add_argument("--history", required=True, type=Path)
    rank_p.add_argument("--current", required=True, type=Path)
    rank_p.add_argument("--changed", required=True, type=Path)
    rank_p.add_argument(
        "--embeddings",
        required=True,
        type=Path,
        help="Directory containing embeddings.npy and test_id_to_idx.csv from Θ_encode",
    )
    rank_p.add_argument(
        "--cluster-mapping",
        required=True,
        type=Path,
        help="CSV from spectral-coarsen: spectral_cluster_mapping.csv",
    )
    rank_p.add_argument("--out", required=True, type=Path)
    rank_p.add_argument("--topk", type=int, default=100)
    rank_p.add_argument("--epochs", type=int, default=100)
    rank_p.set_defaults(func=cmd_rank)

        # spectral-coarsen (Ψ_cluster)
    spec_p = sub.add_parser("spectral-coarsen", help="Run spectral coarsening (Ψ_cluster)")
    spec_p.add_argument(
        "--edges",
        required=True,
        type=Path,
        help="Edges file (CSV or Parquet), typically Ω_prune output",
    )
    spec_p.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output directory for coarsened graph",
    )
    spec_p.add_argument(
        "--k-clusters",
        type=int,
        default=512,
        help="Number of spectral clusters (super-tests)",
    )
    spec_p.set_defaults(func=cmd_spectral_coarsen)
    # compress (Θ_encode)
    
    comp_p = sub.add_parser("compress", help="Run neural coverage compression (Θ_encode)")
    comp_p.add_argument(
        "--edges",
        required=True,
        type=Path,
        help="Edges file after spectral coarsening (CSV or Parquet)",
    )
    comp_p.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output directory for embeddings + stats",
    )
    comp_p.add_argument("--latent-dim", type=int, default=64)
    comp_p.add_argument("--hidden-dim", type=int, default=256)
    comp_p.add_argument("--smooth-lambda", type=float, default=1e-3)
    comp_p.add_argument("--epochs", type=int, default=50)
    comp_p.add_argument("--lr", type=float, default=1e-3)
    comp_p.set_defaults(func=cmd_compress)





    # inspect-data
    inspect_p = sub.add_parser("inspect-data", help="Load and summarize input data")
    inspect_p.add_argument("--edges", required=True, type=Path)
    inspect_p.add_argument("--history", required=True, type=Path)
    inspect_p.add_argument("--current", required=True, type=Path)
    inspect_p.add_argument("--changed", required=True, type=Path)
    inspect_p.set_defaults(func=cmd_inspect_data)

    # reduce (Ω_prune)
    reduce_p = sub.add_parser("reduce", help="Run fault-centric pruning (Ω_prune)")
    reduce_p.add_argument("--edges", required=True, type=Path)
    reduce_p.add_argument("--history", required=True, type=Path)
    reduce_p.add_argument("--current", required=True, type=Path)
    reduce_p.add_argument("--changed", required=True, type=Path)
    reduce_p.add_argument("--out", required=True, type=Path)
    reduce_p.add_argument("--threshold", type=float, default=0.2)
    reduce_p.set_defaults(func=cmd_reduce)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
