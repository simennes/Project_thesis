from __future__ import annotations

import argparse
import json
import os
import logging
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
import tempfile
import pathlib

from src.data import load_data
from src.graph import partition_train_graph, build_global_adjacency
from src.gcn import GCN
from src.utils import set_seed, to_sparse, save_json, _pearson_corr, _select_top_snps_by_abs_corr, _optimizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _merge_best_params(cfg: Dict[str, Any], best_params: Dict[str, Any]) -> Dict[str, Any]:
    """Overlay Optuna best_params into base config (graph/model/training keys)."""
    out = json.loads(json.dumps(cfg))  # deep copy

    # Graph
    g = out.setdefault("graph", {})
    for key in ["knn_k", "weighted_edges", "symmetrize_mode", "laplacian_smoothing", "ensemble_models"]:
        if key in best_params:
            g[key] = best_params[key]

    # Model
    m = out.setdefault("model", {})
    if "hidden_dims" in best_params:
        # best_params["hidden_dims"] may be a JSON string if you encoded choices; accept list as well
        if isinstance(best_params["hidden_dims"], str):
            try:
                m["hidden_dims"] = json.loads(best_params["hidden_dims"])
            except json.JSONDecodeError:
                raise ValueError("hidden_dims in best_params looks like a string but is not valid JSON.")
        else:
            m["hidden_dims"] = best_params["hidden_dims"]
    for key in ["dropout", "batch_norm"]:
        if key in best_params:
            m[key] = best_params[key]

    # Training
    t = out.setdefault("training", {})
    for key in ["lr", "weight_decay", "optimizer", "epochs", "patience"]:
        if key in best_params:
            t[key] = best_params[key]

    # Feature selection (optional)
    fs = out.setdefault("feature_selection", {})
    if "use_snp_selection" in best_params:
        fs["use_snp_selection"] = bool(best_params["use_snp_selection"])
    if "num_snps" in best_params and best_params.get("use_snp_selection", False):
        fs["num_snps"] = int(best_params["num_snps"])

    return out


def train_one_model(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    A_tr_idx, A_tr_val, A_tr_shape,
    X_va: np.ndarray,
    y_va: np.ndarray,
    A_va_idx, A_va_val, A_va_shape,
    X_te: np.ndarray,
    A_te_idx, A_te_val, A_te_shape,
    cfg: Dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    """Train a single GCN or an ensemble on (train, val), return predictions on X_te using A_te*."""
    hidden_dims = cfg["model"]["hidden_dims"]
    dropout = cfg["model"]["dropout"]
    use_bn = cfg["model"]["batch_norm"]

    lr = cfg["training"]["lr"]
    weight_decay = cfg["training"].get("weight_decay", 0.0)
    epochs = int(cfg["training"]["epochs"])
    patience = int(cfg["training"].get("patience", 0))
    opt_name = cfg["training"]["optimizer"].lower()

    ensemble = int(cfg["graph"].get("ensemble_models", 1))
    loss_fn = nn.MSELoss()

    def _new_model(in_dim: int) -> GCN:
        return GCN(in_dim=in_dim, hidden_dims=hidden_dims, dropout=dropout, use_bn=use_bn).to(device)

    if ensemble <= 1:
        model = _new_model(X_tr.shape[1])
        optimizer = _optimizer(opt_name, model.parameters(), lr, weight_decay)

        Xtr_t = torch.from_numpy(X_tr).to(device)
        ytr_t = torch.from_numpy(y_tr).to(device)
        Xva_t = torch.from_numpy(X_va).to(device)

        best_state = None
        best_r = -1.0
        no_imp = 0

        for ep in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            pred = model(Xtr_t, A_tr_idx, A_tr_val, A_tr_shape)
            loss = loss_fn(pred, ytr_t)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                pred_va = model(Xva_t, A_va_idx, A_va_val, A_va_shape).cpu().numpy()
            r = _pearson_corr(y_va, pred_va)
            if r > best_r + 1e-12:
                best_r = r
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
                if patience > 0 and no_imp >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Predict on test fold
        model.eval()
        with torch.no_grad():
            Xte_t = torch.from_numpy(X_te).to(device)
            yhat_te = model(Xte_t, A_te_idx, A_te_val, A_te_shape).cpu().numpy()
        return yhat_te

    # Ensemble: partition train graph and train submodels on disjoint connected subgraphs
    # We’ll slice CSR via indices; build parts from A_train CSR:
    # Rebuild CSR for indices 0..n_tr-1 (already true for A_tr_* tensors), so we can use partition_train_graph.
    from scipy import sparse as sp  # local import to avoid global dependency at import time

    # Convert torch sparse back to CSR just for partitioning convenience
    rows = A_tr_idx[0].cpu().numpy()
    cols = A_tr_idx[1].cpu().numpy()
    data = A_tr_val.detach().cpu().numpy()
    A_train_csr = sp.coo_matrix((data, (rows, cols)), shape=A_tr_shape).tocsr()

    parts = partition_train_graph(A_train_csr, ensemble)

    models: List[GCN] = []
    for nodes in parts:
        if not nodes:
            continue
        A_sub = A_train_csr[nodes][:, nodes].tocsr()
        A_sub_idx, A_sub_val, A_sub_shape = to_sparse(A_sub, device)

        X_sub = X_tr[nodes]
        y_sub = y_tr[nodes]
        X_val = X_va
        y_val = y_va

        model_i = _new_model(X_tr.shape[1])
        optimizer_i = _optimizer(opt_name, model_i.parameters(), lr, weight_decay)

        Xsub_t = torch.from_numpy(X_sub).to(device)
        ysub_t = torch.from_numpy(y_sub).to(device)
        Xval_t = torch.from_numpy(X_val).to(device)

        best_r = -1.0
        best_state = None
        no_imp = 0
        for ep in range(1, epochs + 1):
            model_i.train()
            optimizer_i.zero_grad()
            pred = model_i(Xsub_t, A_sub_idx, A_sub_val, A_sub_shape)
            loss = loss_fn(pred, ysub_t)
            loss.backward()
            optimizer_i.step()

            model_i.eval()
            with torch.no_grad():
                pred_val = model_i(Xval_t, A_va_idx, A_va_val, A_va_shape).cpu().numpy()
            r = _pearson_corr(y_val, pred_val)
            if r > best_r + 1e-12:
                best_r = r
                best_state = {k: v.detach().cpu().clone() for k, v in model_i.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
                if patience > 0 and no_imp >= patience:
                    break

        if best_state is not None:
            model_i.load_state_dict(best_state)
        models.append(model_i)

    # Predict on test fold with mean aggregation
    Xte_t = torch.from_numpy(X_te).to(device)
    preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            preds.append(m(Xte_t, A_te_idx, A_te_val, A_te_shape))
    yhat_te = torch.stack(preds, dim=1).mean(dim=1).cpu().numpy()
    return yhat_te


def run_cv(config_path: str, tuning_results_path: str, n_splits: int = 10) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        base_cfg = json.load(f)

    with open(tuning_results_path, "r", encoding="utf-8") as f:
        tuning_blob = json.load(f)
    best_params = tuning_blob.get("best_params", {})
    cfg = _merge_best_params(base_cfg, best_params)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # Load data
    X, y, ids, GRM_df = load_data(
        cfg["paths"],
        target_column=cfg.get("target_column", "y_adjusted"),
        standardize_features=cfg.get("standardize_features", False),
    )
    n = X.shape[0]
    ids = np.asarray(ids) if ids is not None else np.arange(n)

    A_global = build_global_adjacency(X, GRM_df, cfg["graph"])

    # Prepare CV splitter
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # OOF arrays
    oof_pred = np.zeros(n, dtype=np.float32)
    oof_fold = -np.ones(n, dtype=int)
    fold_metrics: List[Dict[str, float]] = []

    # For early stopping and validation inside each fold, we’ll carve out a val fraction from the train portion
    val_fraction = float(cfg["graph"].get("val_fraction", 0.1))
    if not (0.0 < val_fraction < 1.0):
        val_fraction = 0.1

    # Feature selection settings (graph is still built from ALL SNPs)
    use_sel = bool(cfg.get("feature_selection", {}).get("use_snp_selection", False))
    num_snps = int(cfg.get("feature_selection", {}).get("num_snps", X.shape[1])) if use_sel else X.shape[1]

    for fold_id, (trval_idx, te_idx) in enumerate(kf.split(np.arange(n)), start=1):
        logging.info(f"Fold {fold_id}/{n_splits}: train+val={len(trval_idx)}, test={len(te_idx)}")

        # Induce per-fold train/test graphs from the same global adjacency
        A_test = A_global[te_idx][:, te_idx]

        # Internal split of trval into (train, val) for early stopping
        rng = np.random.default_rng(seed + fold_id)
        perm = rng.permutation(len(trval_idx))
        n_val = max(1, int(len(trval_idx) * val_fraction))
        val_local = trval_idx[perm[:n_val]]
        tr_local = trval_idx[perm[n_val:]]

        # Induce train/val graphs
        A_tr = A_global[tr_local][:, tr_local]
        A_va = A_global[val_local][:, val_local]

        # Feature selection columns computed on TRAIN ONLY (tr_local)
        if use_sel:
            k_cols = min(num_snps, X.shape[1])
            cols_sel = _select_top_snps_by_abs_corr(X[tr_local], y[tr_local], k_cols)
        else:
            cols_sel = slice(None)

        # Convert to torch sparse (once per fold)
        A_tr_idx, A_tr_val, A_tr_shape = to_sparse(A_tr, device)
        A_va_idx, A_va_val, A_va_shape = to_sparse(A_va, device)
        A_te_idx, A_te_val, A_te_shape = to_sparse(A_test, device)

        # Assemble numpy arrays for this fold
        X_tr, y_tr = X[tr_local][:, cols_sel], y[tr_local]
        X_va, y_va = X[val_local][:, cols_sel], y[val_local]
        X_te = X[te_idx][:, cols_sel]

        # Train (single or ensemble) and predict test fold
        yhat_te = train_one_model(
            X_tr, y_tr, A_tr_idx, A_tr_val, A_tr_shape,
            X_va, y_va, A_va_idx, A_va_val, A_va_shape,
            X_te, A_te_idx, A_te_val, A_te_shape,
            cfg, device,
        )

        oof_pred[te_idx] = yhat_te
        oof_fold[te_idx] = fold_id

        # Per-fold metric
        r_fold = _pearson_corr(y[te_idx], yhat_te)
        fold_metrics.append({"fold": fold_id, "pearson_r": r_fold})
        logging.info(f"Fold {fold_id} | OOF Pearson r = {r_fold:.4f}")

    # Overall OOF correlation
    overall_r = _pearson_corr(y, oof_pred)
    logging.info(f"Overall OOF Pearson r (10-fold) = {overall_r:.4f}")

    # Save outputs
    outdir = cfg["paths"]["output_dir"]
    os.makedirs(outdir, exist_ok=True)

    # OOF predictions CSV
    import pandas as pd
    pd.DataFrame(
        {"ringnr": ids, "fold": oof_fold, "y_true": y, "y_pred": oof_pred}
    ).to_csv(os.path.join(outdir, "oof_predictions.csv"), index=False)

    # Metrics JSON
    save_json(
        {
            "overall": {"pearson_r": overall_r},
            "per_fold": fold_metrics,
            "used_params": best_params,
        },
        os.path.join(outdir, "cv_metrics.json"),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="10-fold CV for GCN-RS using tuned hyperparameters (single-config mode)")
    ap.add_argument("--config", type=str, required=True, help="Path to config_validate.json")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        vcfg = json.load(f)

    # allow overriding output_dir
    base_config_path = vcfg["base_config"]
    tuning_results_path = vcfg["tuning_results"]
    n_splits = int(vcfg.get("n_splits", 10))

    # load, then override output dir (optional)
    with open(base_config_path, "r", encoding="utf-8") as f:
        base_cfg = json.load(f)
    if "output_dir" in vcfg:
        base_cfg["paths"]["output_dir"] = vcfg["output_dir"]
    if "seed" in vcfg:
        base_cfg["seed"] = vcfg["seed"]

    # write a temp merged config to pass into run_cv
    tmp_cfg_path = pathlib.Path(tempfile.gettempdir()) / "cv_base_config.json"
    with open(tmp_cfg_path, "w", encoding="utf-8") as tf:
        json.dump(base_cfg, tf)

    run_cv(str(tmp_cfg_path), tuning_results_path, n_splits=n_splits)