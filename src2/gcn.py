import torch
import torch.nn as nn
import torch.nn.functional as F


def spmm(indices, values, shape, dense):
    """
    (Sparse shape[0]xshape[1]) @ dense
    indices: (2, nnz), values: (nnz,), dense: (shape[1], d)
    """
    A = torch.sparse_coo_tensor(indices, values, torch.Size(shape)).coalesce()
    return torch.sparse.mm(A, dense)


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0, use_bn=False):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(out_dim) if use_bn else None

    def forward(self, x, A_indices, A_values, A_shape):
        h = self.lin(x)
        h = spmm(A_indices, A_values, A_shape, h) + self.bias
        if self.bn is not None:
            h = self.bn(h)
        return F.relu(self.dropout(h))


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dims, dropout=0.0, use_bn=False):
        super().__init__()
        dims = [in_dim] + list(hidden_dims)
        self.layers = nn.ModuleList(
            [GCNLayer(dims[i], dims[i + 1], dropout=dropout, use_bn=use_bn) for i in range(len(dims) - 1)]
        )
        self.out = nn.Linear(dims[-1], 1)

    def forward(self, x, A_indices, A_values, A_shape):
        h = x
        for layer in self.layers:
            h = layer(h, A_indices, A_values, A_shape)
        y = self.out(h).squeeze(-1)
        return y
