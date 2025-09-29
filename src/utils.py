import os
import json
import numpy as np
import torch
from torch.optim import Adam, SGD
from typing import Any, List

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_torch_sparse(A_csr) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    A_coo = A_csr.tocoo()
    idx_np = np.vstack((A_coo.row, A_coo.col))  # shape (2, nnz)
    indices = torch.from_numpy(idx_np).long()
    values = torch.from_numpy(A_coo.data).float()
    shape = (A_coo.shape[0], A_coo.shape[1])
    return indices, values, shape

# Convert to torch sparse
def to_sparse(A, device):
    idx, val, shape = to_torch_sparse(A)
    return idx.to(device), val.to(device), shape

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    mse = ((y_true - y_pred) ** 2).mean()
    rmse = float(np.sqrt(mse))
    mae = float(np.abs(y_true - y_pred).mean())
    # r2 can be negative if model is worse than mean prediction
    denom = ((y_true - y_true.mean()) ** 2).sum()
    r2 = float(1.0 - ((y_true - y_pred) ** 2).sum() / denom) if denom > 0 else float("nan")
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    return {"rmse": rmse, "mae": mae, "r2": r2, "corr": corr}

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _split_indices(n: int, val_fraction: float, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return train_idx, val_idx, test_idx


def _optimizer(name: str, params, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay)
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