from __future__ import annotations
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
from torch_geometric.utils import k_hop_subgraph
from typing import Any, Dict, List, Optional


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_torch_sparse(A_csr) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    A_coo = A_csr.tocoo()
    idx_np = np.vstack((A_coo.row, A_coo.col))
    indices = torch.from_numpy(idx_np).long()
    values = torch.from_numpy(A_coo.data).float()
    shape = (A_coo.shape[0], A_coo.shape[1])
    return indices, values, shape


def to_sparse(A, device):
    """Convert CSR matrix to edge_index, edge_weight tensors on device."""
    idx, val, shape = to_torch_sparse(A)
    return idx.to(device), val.to(device), shape


def _optimizer(name: str, params, lr: float, weight_decay: float):
    name = (name or "adam").lower()
    if name == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")

def _select_top_snps_by_abs_corr(X_train: np.ndarray, y_train: np.ndarray, k: int) -> np.ndarray:
    """
    Return indices of top-k SNPs by absolute Pearson correlation with y_train.
    Works on TRAIN ONLY to avoid leakage.
    """
    k = int(k)
    p = X_train.shape[1]
    if k <= 0 or k >= p:
        return np.arange(p, dtype=int)

    y = y_train.astype(np.float64)
    y = y - y.mean()
    y_norm = np.sqrt((y * y).sum())
    if y_norm == 0.0:
        # y is constant; fall back to variance ranking
        x_var = X_train.var(axis=0)
        return np.argpartition(-x_var, kth=min(k, p - 1))[:k]

    Xc = X_train.astype(np.float64) - X_train.mean(axis=0, keepdims=True)
    num = (Xc * y[:, None]).sum(axis=0)
    x_norm = np.sqrt((Xc * Xc).sum(axis=0))
    denom = x_norm * (y_norm + 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0.0, num / denom, 0.0)
    score = np.abs(corr)
    if k < p:
        idx = np.argpartition(-score, kth=k - 1)[:k]
    else:
        idx = np.arange(p)
    # (optional) sort for consistent column order
    return idx[np.argsort(-score[idx])]

def _pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return Pearson r; if constant variance, return 0.0 to be safe."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0 or y_pred.size == 0:
        return 0.0
    yt = y_true - y_true.mean()
    yp = y_pred - y_pred.mean()
    denom = np.sqrt((yt * yt).sum()) * np.sqrt((yp * yp).sum())
    if denom == 0.0:
        return 0.0
    return float((yt * yp).sum() / denom)

def encode_choices_for_optuna(choices: List[Any]) -> List[str]:
    """
    Encode list-of-lists (or other non-primitive values) into JSON strings
    so they can be passed safely to Optuna's suggest_categorical without warnings.
    """
    return [json.dumps(c) for c in choices]


def decode_choice(choice: str) -> Any:
    """
    Decode a JSON string sampled from Optuna back to its original object.
    """
    return json.loads(choice)


# ---------------------- Training utilities ----------------------

def make_loss(name: str):
    """Create loss function by name."""
    name = (name or "mse").lower()
    return nn.L1Loss() if name == "mae" else nn.MSELoss()


def train_masked_epochs(model: nn.Module,
                        x_all: torch.Tensor,
                        edge_index: torch.Tensor,
                        edge_weight: Optional[torch.Tensor],
                        y_all: torch.Tensor,
                        train_idx: np.ndarray,
                        epochs: int,
                        opt: torch.optim.Optimizer,
                        loss_fn: nn.Module):
    """Transductive: forward on **all nodes**, compute loss only on train_idx."""
    tr_idx_t = torch.tensor(train_idx, dtype=torch.long, device=x_all.device)
    for _ in range(int(epochs)):
        model.train()
        opt.zero_grad()
        preds = model(x_all, edge_index, edge_weight)
        loss = loss_fn(preds.index_select(0, tr_idx_t), y_all.index_select(0, tr_idx_t))
        loss.backward()
        opt.step()


def _resolve_graphsage_num_hops(hidden_dims: Optional[List[int]], override: Optional[int]) -> int:
    """Determine number of hops for GraphSAGE sampling based on hidden layers."""
    if override is not None:
        return max(1, int(override))
    if hidden_dims is None:
        return 1
    return max(1, len(hidden_dims))


def train_graphsage_minibatches(model: nn.Module,
                                opt: torch.optim.Optimizer,
                                loss_fn: nn.Module,
                                graphs: List[Dict[str, Any]],
                                epochs: int,
                                batch_size: int,
                                num_hops: int,
                                shuffle_nodes: bool = True,
                                drop_last: bool = False):
    """Train GraphSAGE model with minibatch sampling across multiple graphs."""
    if not graphs:
        return
    batch_size = max(1, int(batch_size))
    num_hops = max(1, int(num_hops))
    for _ in range(int(epochs)):
        valid_graphs = []
        max_batches = 0
        for gdat in graphs:
            x = gdat["x"]
            n_nodes = int(x.shape[0])
            if n_nodes == 0:
                continue
            if shuffle_nodes:
                node_order = torch.randperm(n_nodes, device=x.device)
            else:
                node_order = torch.arange(n_nodes, device=x.device)
            batches = (n_nodes + batch_size - 1) // batch_size
            max_batches = max(max_batches, batches)
            valid_graphs.append({
                "gdat": gdat,
                "n_nodes": n_nodes,
                "node_order": node_order,
            })

        if not valid_graphs or max_batches == 0:
            continue

        for bidx in range(max_batches):
            for entry in valid_graphs:
                gdat = entry["gdat"]
                n_nodes = entry["n_nodes"]
                node_order = entry["node_order"]
                start = bidx * batch_size
                if start >= n_nodes:
                    continue
                end = min(start + batch_size, n_nodes)
                if drop_last and (end - start) < batch_size:
                    continue
                seed_nodes = node_order[start:end]
                if seed_nodes.numel() == 0:
                    continue

                edge_index = gdat["edge_index"]
                edge_weight = gdat.get("edge_weight")
                x = gdat["x"]
                y = gdat["y"]

                subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
                    seed_nodes,
                    num_hops,
                    edge_index,
                    relabel_nodes=True,
                    num_nodes=n_nodes,
                )

                if subset.numel() == 0 or mapping.numel() == 0:
                    continue

                subset = subset.long()
                mapping = mapping.long()
                seed_nodes_long = seed_nodes.long()
                if edge_mask is not None and edge_mask.device != edge_index.device:
                    edge_mask = edge_mask.to(edge_index.device)

                x_sub = x.index_select(0, subset)
                y_target = y.index_select(0, seed_nodes_long)

                edge_weight_sub = None
                if edge_weight is not None and edge_weight.numel() > 0 and edge_mask is not None:
                    if edge_mask.dtype == torch.bool:
                        edge_weight_sub = edge_weight[edge_mask]
                    else:
                        edge_weight_sub = edge_weight.index_select(0, edge_mask.long())

                model.train()
                opt.zero_grad()
                pred_sub = model(x_sub, sub_edge_index, edge_weight_sub)
                pred_seed = pred_sub.index_select(0, mapping)
                loss = loss_fn(pred_seed, y_target)
                loss.backward()
                opt.step()