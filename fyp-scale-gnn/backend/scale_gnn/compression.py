from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from rich.console import Console

console = Console()


@dataclass
class CompressionStats:
    strategy_name: str
    strategy_params: Dict[str, float]

    num_tests: int
    num_files: int
    input_dim: int
    latent_dim: int
    reduction_ratio: float  # latent_dim / input_dim

    num_edges_tt: int  # test–test edges used 
    smooth_lambda: float
    epochs: int
    train_loss: float


class GraphAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


def _build_incidence(edges: pd.DataFrame) -> Tuple[sp.csr_matrix, Dict[str, int], Dict[str, int]]:
    """
    Build tests×files incidence matrix B from edges[test_id, src_file].
    """
    test_ids = sorted(edges["test_id"].unique())
    file_ids = sorted(edges["src_file"].unique())
    test2idx = {t: i for i, t in enumerate(test_ids)}
    file2idx = {f: i for i, f in enumerate(file_ids)}

    row = [test2idx[t] for t in edges["test_id"]]
    col = [file2idx[f] for f in edges["src_file"]]
    data = np.ones(len(row), dtype=np.float32)

    B = sp.coo_matrix((data, (row, col)), shape=(len(test_ids), len(file_ids))).tocsr()
    return B, test2idx, file2idx


def neural_compress_coverage(
    edges: pd.DataFrame,
    latent_dim: int = 64,
    hidden_dim: int = 256,
    smooth_lambda: float = 1e-3,
    epochs: int = 50,
    lr: float = 1e-3,
    strategy_name: str = "graph_autoencoder_compress",
    device: str = "cuda_if_available",
) -> Tuple[np.ndarray, Dict[str, int], CompressionStats]:
    """
    Θ_encode: Graph-aware neural compression over test×file coverage.

    - x = incidence matrix B (dense)
    - Build test–test adjacency A = B B^T (co-occurrence in files)
    - Train an autoencoder with:
        loss = MSE(x_hat, x) + λ * sum_{(i,j) in E_tt} ||z_i - z_j||^2

    Returns:
      embeddings (np.ndarray [num_tests, latent_dim]),
      test2idx mapping,
      CompressionStats
    """
    console.log("[Θ_encode] building incidence matrix...")
    B, test2idx, file2idx = _build_incidence(edges)
    num_tests, num_files = B.shape

    console.log(f"[Θ_encode] num_tests={num_tests}, num_files={num_files}")

    # Dense input features: each test is a file-coverage vector
    x_np = B.toarray().astype(np.float32)
    x = torch.from_numpy(x_np)

    # Test–test adjacency (co-occurrence)
    console.log("[Θ_encode] building test–test adjacency...")
    A = B @ B.T
    A.setdiag(0)
    A.eliminate_zeros()

    tt_coo = A.tocoo()
    edge_i = torch.from_numpy(tt_coo.row.astype(np.int64))
    edge_j = torch.from_numpy(tt_coo.col.astype(np.int64))
    num_edges_tt = edge_i.numel()
    console.log(f"[Θ_encode] test–test edges (non-zero co-coverage pairs): {num_edges_tt}")

    # Device
    if device == "cuda_if_available" and torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    x = x.to(dev)
    edge_i = edge_i.to(dev)
    edge_j = edge_j.to(dev)

    model = GraphAutoEncoder(input_dim=num_files, latent_dim=latent_dim, hidden_dim=hidden_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    console.log(f"[Θ_encode] training autoencoder for {epochs} epochs on {dev}...")
    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        x_hat, z = model(x)

        recon_loss = mse(x_hat, x)

        if num_edges_tt > 0:
            zi = z[edge_i]
            zj = z[edge_j]
            smooth_loss = ((zi - zj) ** 2).sum() / num_edges_tt
        else:
            smooth_loss = torch.tensor(0.0, device=dev)

        loss = recon_loss + smooth_lambda * smooth_loss
        loss.backward()
        opt.step()

        if epoch % 10 == 0 or epoch == epochs - 1:
            console.log(
                f"[Θ_encode] epoch {epoch+1}/{epochs} "
                f"recon={recon_loss.item():.4f} smooth={smooth_loss.item():.4f} "
                f"total={loss.item():.4f}"
            )

    model.eval()
    with torch.no_grad():
        _, z = model(x)
    z_np = z.cpu().numpy()

    reduction_ratio = latent_dim / float(num_files)

    stats = CompressionStats(
        strategy_name=strategy_name,
        strategy_params={
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "smooth_lambda": smooth_lambda,
            "epochs": epochs,
            "lr": lr,
        },
        num_tests=num_tests,
        num_files=num_files,
        input_dim=num_files,
        latent_dim=latent_dim,
        reduction_ratio=reduction_ratio,
        num_edges_tt=int(num_edges_tt),
        smooth_lambda=smooth_lambda,
        epochs=epochs,
        train_loss=float(loss.item()),
    )

    return z_np, test2idx, stats


def save_embeddings_and_stats(
    embeddings: np.ndarray,
    test2idx: Dict[str, int],
    stats: CompressionStats,
    out_dir: Path,
) -> None:
    """
    Save:
      - embeddings.npy
      - test_id_to_idx.csv
      - compression_stats.json
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path = out_dir / "embeddings.npy"
    np.save(emb_path, embeddings)

    # mapping
    inv = {idx: tid for tid, idx in test2idx.items()}
    rows = [{"test_id": inv[i], "idx": i} for i in range(len(inv))]
    map_df = pd.DataFrame(rows)
    map_path = out_dir / "test_id_to_idx.csv"
    map_df.to_csv(map_path, index=False)

    # stats
    import json

    stats_path = out_dir / "compression_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(stats), f, indent=2)

    console.log(f"[Θ_encode] wrote embeddings to {emb_path}")
    console.log(f"[Θ_encode] wrote test-id mapping to {map_path}")
    console.log(f"[Θ_encode] wrote compression stats to {stats_path}")
