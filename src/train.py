# src/train.py
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD

from data import load_data
from graph import build_knn_from_grm, gcn_normalize, sample_subgraph_indices
from gcn import GCN
from utils import set_seed, to_torch_sparse, metrics, save_json


def split_indices(n, val_fraction, test_fraction, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return train_idx, val_idx, test_idx


def train_one_subgraph(model, optimizer, loss_fn,
                       X, y, A_idx, A_val, A_shape,
                       sub_idx, device):
    model.train()
    xs = torch.from_numpy(X[sub_idx]).to(device)
    ys = torch.from_numpy(y[sub_idx]).to(device)

    # remap adjacency to subgraph (induced)
    # Build mask & mapping
    sub_mask = np.zeros(A_shape[0], dtype=bool)
    sub_mask[sub_idx] = True
    # Extract rows/cols in subgraph
    rows = A_idx[0].cpu().numpy()
    cols = A_idx[1].cpu().numpy()
    keep = sub_mask[rows] & sub_mask[cols]
    sub_rows = rows[keep]
    sub_cols = cols[keep]
    # Remap to 0..m-1
    remap = -np.ones(A_shape[0], dtype=int)
    remap[sub_idx] = np.arange(len(sub_idx))
    sub_rows = remap[sub_rows]
    sub_cols = remap[sub_cols]
    sub_vals = A_val.cpu().numpy()[keep]
    sub_indices = torch.tensor([sub_rows, sub_cols], dtype=torch.long, device=device)
    sub_values = torch.tensor(sub_vals, dtype=torch.float32, device=device)
    sub_shape = (len(sub_idx), len(sub_idx))

    optimizer.zero_grad()
    pred = model(xs, sub_indices, sub_values, sub_shape)
    loss = loss_fn(pred, ys)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def evaluate_full(model, X, y, A_idx, A_val, A_shape, eval_idx, device):
    model.eval()
    with torch.no_grad():
        x_eval = torch.from_numpy(X).to(device)
        y_true = y[eval_idx]
        pred_all = model(x_eval, A_idx, A_val, A_shape).cpu().numpy()
        y_pred = pred_all[eval_idx]
    return metrics(y_true, y_pred), y_pred


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"])

    # --- Load data ---
    X, y, ids, GRM_df = load_data(cfg["paths"],
                                  target_column=cfg.get("target_column", "y_adjusted"),
                                  standardize_features=cfg.get("standardize_features", False))

    # --- Build graph from GRM ---
    A = build_knn_from_grm(
        GRM_df,
        k=cfg["graph"]["knn_k"],
        weighted_edges=cfg["graph"]["weighted_edges"],
        symmetrize_mode=cfg["graph"].get("symmetrize_mode", "union"),
        add_self_loops=cfg["graph"].get("self_loops", True),
    )
    A_norm = gcn_normalize(A)
    A_idx, A_val, A_shape = to_torch_sparse(A_norm)
    A_idx = A_idx.to(device)
    A_val = A_val.to(device)

    # --- Splits ---
    n = X.shape[0]
    train_idx, val_idx, test_idx = split_indices(
        n,
        cfg["training"]["val_fraction"],
        cfg["training"]["test_fraction"],
        seed=cfg["seed"]
    )

    # --- Model ---
    model = GCN(
        in_dim=X.shape[1],
        hidden_dims=cfg["model"]["hidden_dims"],
        dropout=cfg["model"]["dropout"],
        use_bn=cfg["model"]["batch_norm"]
    ).to(device)

    # --- Optimizer ---
    opt_name = cfg["training"]["optimizer"].lower()
    lr = cfg["training"]["lr"]
    weight_decay = cfg["training"].get("weight_decay", 0.0)
    if opt_name == "adam":
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    loss_fn = nn.MSELoss()

    # --- Training loop with subgraph sampling ---
    rng = np.random.default_rng(cfg["seed"])
    frac = cfg["graph"]["subgraph_size_fraction"]
    subgraphs_per_epoch = cfg["graph"]["num_subgraphs_per_epoch"]

    history = {"epoch": [], "train_loss": [], "val_rmse": [], "val_mae": [], "val_r2": []}

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        epoch_losses = []
        for _ in range(subgraphs_per_epoch):
            sub_idx = sample_subgraph_indices(train_idx, frac, rng)
            loss = train_one_subgraph(
                model, optimizer, loss_fn,
                X, y, A_idx, A_val, A_shape,
                sub_idx, device
            )
            epoch_losses.append(loss)

        # Validate on full graph (val split only)
        val_metrics, _ = evaluate_full(model, X, y, A_idx, A_val, A_shape, val_idx, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(float(np.mean(epoch_losses)))
        history["val_rmse"].append(val_metrics["rmse"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_r2"].append(val_metrics["r2"])

        if epoch % max(1, cfg["training"].get("log_every", 10)) == 0:
            print(f"[{epoch:03d}] train_loss={np.mean(epoch_losses):.4f} "
                  f"val_rmse={val_metrics['rmse']:.4f} r2={val_metrics['r2']:.4f}")

    # Final test evaluation
    test_metrics, test_pred = evaluate_full(model, X, y, A_idx, A_val, A_shape, test_idx, device)
    print("TEST:", test_metrics)

    # Save artifacts
    outdir = cfg["paths"]["output_dir"]
    os.makedirs(outdir, exist_ok=True)
    # Save metrics
    run_summary = {
        "config": cfg,
        "val_last": {k: history[k][-1] for k in ["val_rmse", "val_mae", "val_r2"]},
        "test": test_metrics
    }
    save_json(run_summary, os.path.join(outdir, "summary.json"))
    # Save per-epoch
    import pandas as pd
    pd.DataFrame(history).to_csv(os.path.join(outdir, "metrics.csv"), index=False)
    # Save predictions
    pd.DataFrame({
        "ringnr": ids[test_idx],
        "y_true": y[test_idx],
        "y_pred": test_pred
    }).to_csv(os.path.join(outdir, "test_predictions.csv"), index=False)
    # Save model
    torch.save(model.state_dict(), os.path.join(outdir, "model.pt"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to config_train.json")
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = json.load(f)
    main(cfg)
