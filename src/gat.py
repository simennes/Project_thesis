import torch
import torch.nn as nn
from typing import List, Optional, Union
from torch_geometric.nn import GATConv


class PyGGAT(nn.Module):
    """Multi-layer GAT with optional multi-head attention per hidden layer.

    Notes:
    - For hidden layers, if heads > 1 and concat=True, the next layer's input dim is out_dim * heads.
    - The final prediction is produced by a Linear layer to shape (n, 1), independent of heads.
    - edge_weight is accepted for API compatibility but ignored by GATConv.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.0,
        batch_norm: bool = False,
        heads: Union[int, List[int]] = 1,
        attn_dropout: float = 0.0,
        concat_hidden: bool = True,
    ):
        super().__init__()
        # normalize heads to per-layer list
        if isinstance(heads, int):
            heads_list = [heads] * len(hidden_dims)
        else:
            heads_list = list(heads)
            if len(heads_list) != len(hidden_dims):
                # pad or trim to match number of layers
                if len(heads_list) < len(hidden_dims):
                    heads_list = heads_list + [heads_list[-1]] * (len(hidden_dims) - len(heads_list))
                else:
                    heads_list = heads_list[: len(hidden_dims)]

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = batch_norm
        self.concat_hidden = bool(concat_hidden)
        self.act = nn.ReLU()

        prev_dim = in_dim
        for i, out_dim in enumerate(hidden_dims):
            h = int(max(1, heads_list[i]))
            # Whether to concatenate heads for this layer's output
            concat_here = bool(self.concat_hidden)
            self.convs.append(
                GATConv(
                    prev_dim,
                    out_dim,
                    heads=h,
                    dropout=float(attn_dropout),
                    concat=concat_here,
                )
            )
            bn_dim = (out_dim * h) if concat_here else out_dim
            self.bns.append(nn.BatchNorm1d(bn_dim))
            # Update prev_dim for next layer to match the actual output dimension
            prev_dim = bn_dim

        self.out_lin = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.batch_norm:
                x = self.bns[i](x)
            x = self.act(x)
            x = self.dropout(x)
        out = self.out_lin(x).squeeze(-1)
        return out
