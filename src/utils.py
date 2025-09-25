# src/utils.py
import os
import json
import numpy as np
import torch


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_torch_sparse(A_csr):
    A_coo = A_csr.tocoo()
    idx_np = np.vstack((A_coo.row, A_coo.col))  # shape (2, nnz)
    indices = torch.from_numpy(idx_np).long()
    values = torch.from_numpy(A_coo.data).float()
    shape = (A_coo.shape[0], A_coo.shape[1])
    return indices, values, shape


def metrics(y_true, y_pred):
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    mse = ((y_true - y_pred) ** 2).mean()
    rmse = float(np.sqrt(mse))
    mae = float(np.abs(y_true - y_pred).mean())
    # r2 can be negative
    denom = ((y_true - y_true.mean()) ** 2).sum()
    r2 = float(1.0 - ((y_true - y_pred) ** 2).sum() / denom) if denom > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
