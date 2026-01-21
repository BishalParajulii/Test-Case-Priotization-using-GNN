# scale_gnn/model.py
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class ScaleGCN(nn.Module):
    """
    Simple per-node GCN for test risk prediction.

    Input: x (features per test)
    Output: logit per test (before sigmoid)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = dropout
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        logits = self.fc_out(x).view(-1)  # one logit per test
        return logits


def train_scale_gcn(
    model: ScaleGCN,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    num_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
) -> ScaleGCN:
    """
    Simple full-batch training on all nodes.
    y is binary {0,1} per test.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)
    y = y.to(device)

    # Handle class imbalance
    pos = (y == 1).sum().item()
    neg = (y == 0).sum().item()
    if pos == 0 or neg == 0:
        pos_weight = torch.tensor(1.0, device=device)
    else:
        pos_weight = torch.tensor(neg / max(pos, 1), device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = criterion(logits, y.float())
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long()
                acc = (preds == y).float().mean().item()
            print(
                f"[ScaleGCN] epoch {epoch+1}/{num_epochs}, "
                f"loss={loss.item():.4f}, acc={acc:.4f}, pos_weight={pos_weight.item():.3f}"
            )

    return model


def predict_scale_gcn(
    model: ScaleGCN,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Returns probabilities (sigmoid logits) per node/test.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()
    x = x.to(device)
    edge_index = edge_index.to(device)

    with torch.no_grad():
        logits = model(x, edge_index)
        probs = torch.sigmoid(logits)

    return probs.cpu()
