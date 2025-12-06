import torch
import torch.nn as nn
from typing import List, Optional


class FeedForwardMLP(nn.Module):
    """Simple fully connected regressor matching the GNN interface."""

    def __init__(self, in_dim: int, hidden_dims: Optional[List[int]] = None, dropout: float = 0.0, batch_norm: bool = False):
        super().__init__()
        hidden_dims = hidden_dims or []
        dims = [in_dim] + list(hidden_dims)
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = batch_norm
        self.act = nn.ReLU()

        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.bns.append(nn.BatchNorm1d(dims[i + 1]))

        last_dim = dims[-1]
        self.out_lin = nn.Linear(last_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None, edge_weight: Optional[torch.Tensor] = None):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if self.batch_norm:
                h = self.bns[i](h)
            h = self.act(h)
            h = self.dropout(h)
        out = self.out_lin(h).squeeze(-1)
        return out
