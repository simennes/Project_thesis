from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
from typing import List, Optional
from dataclasses import dataclass


class PyGGCN(nn.Module):
    """Simple MLP+GCN stack via PyTorch Geometric.

    Accepts dense X (n×p) and either:
      - PyG graph (edge_index, edge_weight)
    We keep it small and dependable; extend with batch norm/residuals as needed.
    """
    def __init__(self, in_dim: int, hidden_dims: List[int], dropout: float = 0.0, batch_norm: bool = False):
        super().__init__()
        dims = [in_dim] + hidden_dims + [1]
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = batch_norm

        for i in range(len(dims) - 2):
            self.convs.append(GCNConv(dims[i], dims[i+1], normalize=True))
            self.bns.append(nn.BatchNorm1d(dims[i+1]))
        self.out_lin = nn.Linear(dims[-2], dims[-1])
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if self.batch_norm:
                x = self.bns[i](x)
            x = self.act(x)
            x = self.dropout(x)
        out = self.out_lin(x).squeeze(-1)
        return out
    
@dataclass
class TrainParams:
    lr: float
    weight_decay: float
    epochs: int
    loss_name: str
    optimizer: str
    hidden_dims: List[int]
    dropout: float
    batch_norm: bool
    
def make_model(in_dim: int, tp: TrainParams) -> nn.Module:
    return PyGGCN(in_dim=in_dim, hidden_dims=tp.hidden_dims, dropout=tp.dropout, batch_norm=tp.batch_norm)