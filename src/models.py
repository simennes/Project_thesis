from dataclasses import dataclass
from typing import List, Optional, Union
import torch.nn as nn

from .gcn import PyGGCN
from .gat import PyGGAT
from .graphsage import PyGSAGE


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
    # Optional model-specific params
    model_type: str = "gcn"  # gcn | gat | graphsage
    gat_heads: Optional[int] = None
    gat_attn_dropout: Optional[float] = None
    gat_concat_hidden: Optional[bool] = None


def make_model(in_dim: int, tp: TrainParams) -> nn.Module:
    mtype = (tp.model_type or "gcn").lower()
    if mtype == "gcn":
        return PyGGCN(in_dim=in_dim, hidden_dims=tp.hidden_dims, dropout=tp.dropout, batch_norm=tp.batch_norm)
    if mtype == "gat":
        heads = tp.gat_heads if tp.gat_heads is not None else 1
        attn_dropout = tp.gat_attn_dropout if tp.gat_attn_dropout is not None else 0.0
        concat_hidden = tp.gat_concat_hidden if tp.gat_concat_hidden is not None else True
        return PyGGAT(
            in_dim=in_dim,
            hidden_dims=tp.hidden_dims,
            dropout=tp.dropout,
            batch_norm=tp.batch_norm,
            heads=heads,
            attn_dropout=attn_dropout,
            concat_hidden=concat_hidden,
        )
    if mtype == "graphsage":
        return PyGSAGE(in_dim=in_dim, hidden_dims=tp.hidden_dims, dropout=tp.dropout, batch_norm=tp.batch_norm)
    # default fallback
    return PyGGCN(in_dim=in_dim, hidden_dims=tp.hidden_dims, dropout=tp.dropout, batch_norm=tp.batch_norm)
