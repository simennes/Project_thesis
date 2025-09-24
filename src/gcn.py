# src/gcn.py
import torch
import torch.nn as nn
from typing import List


def _spmm(indices, values, m, n, dense):
    """
    Sparse-dense matmul: (sparse m x n) @ (dense n x d) -> (m x d)
    indices: 2 x nnz LongTensor
    values: nnz FloatTensor
    dense: n x d FloatTensor
    """
    A = torch.sparse_coo_tensor(indices, values, (m, n))
    return torch.sparse.mm(A, dense)


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0, use_bn=False):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim) if use_bn else None
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, A_indices, A_values, shape):
        m, n = shape
        h = _spmm(A_indices, A_values, m, n, x)
        h = self.linear(h)
        if self.bn is not None:
            h = self.bn(h)
        h = self.act(h)
        h = self.drop(h)
        return h


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], dropout=0.0, use_bn=True):
        super().__init__()
        dims = [in_dim] + list(hidden_dims) + [1]  # regression head
        layers = []
        for i in range(len(dims) - 2):  # hidden layers
            layers.append(GCNLayer(dims[i], dims[i + 1], dropout=dropout, use_bn=use_bn))
        self.layers = nn.ModuleList(layers)
        self.out = nn.Linear(dims[-2], dims[-1], bias=True)

    def forward(self, x, A_indices, A_values, shape):
        h = x
        for layer in self.layers:
            h = layer(h, A_indices, A_values, shape)
        y = self.out(h).squeeze(-1)
        return y
