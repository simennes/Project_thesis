import torch
import torch.nn as nn
from typing import List, Optional
from torch_geometric.nn import SAGEConv


class PyGSAGE(nn.Module):
    """Multi-layer GraphSAGE model (mean aggregator)."""

    def __init__(self, in_dim: int, hidden_dims: List[int], dropout: float = 0.0, batch_norm: bool = False):
        super().__init__()
        dims = [in_dim] + hidden_dims
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = batch_norm
        self.act = nn.ReLU()

        for i in range(len(dims) - 1):
            self.convs.append(SAGEConv(dims[i], dims[i + 1]))
            self.bns.append(nn.BatchNorm1d(dims[i + 1]))
        self.out_lin = nn.Linear(dims[-1], 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.batch_norm:
                x = self.bns[i](x)
            x = self.act(x)
            x = self.dropout(x)
        out = self.out_lin(x).squeeze(-1)
        return out
