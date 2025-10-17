from __future__ import annotations
import argparse
import json
import logging
import os
import random
from typing import Dict, Tuple, Optional, List

import numpy as np
import optuna
from optuna.pruners import MedianPruner, NopPruner
from optuna.samplers import TPESampler

import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import scipy.sparse as sp

from src.data import load_data           # <-- your loader
from src2.graph import build_global_adjacency
from src2.gcn import GCN
from src.utils import encode_choices_for_optuna, decode_choice


# -------------------- utils --------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_torch(x: np.ndarray, device: torch.device):
    return torch.from_numpy(x).to(device)


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(pearsonr(a, b)[0])


def top_k_by_abs_corr(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    y_c = y - y.mean()
    y_ss = np.sum(y_c ** 2)
    if y_ss <= 0:
        vari = np.var(X, axis=0)
        return np.argsort(vari)[::-1][:k]
    X_c = X - X.mean(axis=0)
    num = np.abs(np.dot(X_c.T, y_c))
    den = np.sqrt(np.sum(X_c**2, axis=0) * y_ss) + 1e-12
    r = num / den
    return np.argsort(r)[::-1][:k]


def try_load_eval_from_npz(paths_cfg: Dict, eval_col: str) -> Optional[np.ndarray]:
    """
    If an NPZ exists and contains eval_col, return it; else None.
    Accepts 'npz' or 'npz_path' inside paths.
    """
    npz_key = "npz" if "npz" in paths_cfg else ("npz_path" if "npz_path" in paths_cfg else None)
    if not npz_key:
        return None
    npz_path = paths_cfg.get(npz_key)
    if not npz_path or not os.path.exists(npz_path):
        return None
    data = np.load(npz_path, allow_pickle=False)
    if eval_col in data.files:
        return data[eval_col].astype(np.float32, copy=False)
    # historical alias used earlier
    if eval_col == "y_mean" and "y_eval_target" in data.files:
        return data["y_eval_target"].astype(np.float32, copy=False)
    return None


# -------------------- training loops --------------------
def train_one(
    model: nn.Module,
    X: torch.Tensor,
    A_indices: torch.Tensor,
    A_values: torch.Tensor,
    A_shape: Tuple[int, int],
    y: torch.Tensor,
    train_mask: np.ndarray,
    lr: float,
    weight_decay: float,
    epochs: int,
    device: torch.device,
    val_mask: Optional[np.ndarray] = None,
    trial: Optional[optuna.Trial] = None,
    prune_warmup_epochs: int = 5,
    prune_interval: int = 1,
    report_offset: int = 0,
    enable_pruning: bool = True,
) -> nn.Module:
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    tr_idx = torch.from_numpy(np.where(train_mask)[0]).long().to(device)
    mse = nn.MSELoss()
    y_cpu = y.detach().cpu().numpy()

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        pred = model(X, A_indices, A_values, A_shape)
        loss = mse(pred[tr_idx], y[tr_idx])
        loss.backward()
        opt.step()

        # Pruner reporting on validation metric (optional)
        if enable_pruning and trial is not None and val_mask is not None:
            model.eval()
            with torch.no_grad():
                pred_all = model(X, A_indices, A_values, A_shape).detach().cpu().numpy()
            r_val = pearson_corr(pred_all[val_mask], y_cpu[val_mask])
            # Use unique step numbers to avoid duplicate warnings across inner folds
            trial.report(r_val, step=report_offset + epoch)
            if epoch >= prune_warmup_epochs and (epoch % prune_interval == 0):
                if trial.should_prune():
                    raise optuna.TrialPruned()
        
    return model


def train_full(
    model: nn.Module,
    X: torch.Tensor,
    A_indices: torch.Tensor,
    A_values: torch.Tensor,
    A_shape: Tuple[int, int],
    y: torch.Tensor,
    train_mask: np.ndarray,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    device: torch.device,
):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    tr_idx = torch.from_numpy(np.where(train_mask)[0]).long().to(device)
    mse = nn.MSELoss()
    for _ in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad()
        pred = model(X, A_indices, A_values, A_shape)
        loss = mse(pred[tr_idx], y[tr_idx])
        loss.backward()
        opt.step()


# -------------------- objective factory (transductive) --------------------
def objective_factory(
    X_all: np.ndarray,
    y_all: np.ndarray,
    GRM_mat: Optional[np.ndarray],
    outer_train_idx: np.ndarray,
    cfg: Dict,
    search_space: Dict,
    device: torch.device,
):
    k_inner = int(cfg.get("inner_folds", 5))
    seed = int(cfg.get("base_train", {}).get("seed", 42))
    kf = KFold(n_splits=k_inner, shuffle=True, random_state=seed)
    inner_splits = list(kf.split(outer_train_idx))

    def objective(trial: optuna.Trial) -> float:
        # ---- model hyperparams
        mspace = search_space.get("model", {})
        hd_choices = mspace.get("hidden_dims_choices", [[128, 64], [256, 128], [64, 64]])
        hd_choices_str = encode_choices_for_optuna(hd_choices)
        hidden_dims = trial.suggest_categorical("hidden_dims", hd_choices_str)
        hidden_dims = decode_choice(hidden_dims)
        dropout = trial.suggest_float("dropout", *mspace.get("dropout_range", (0.0, 0.6)))
        use_bn = trial.suggest_categorical("batch_norm", mspace.get("batch_norm_choices", [True, False]))

        # ---- graph hyperparams with explicit toggle
        gspace = search_space.get("graph", {})
        graph_on = trial.suggest_categorical("graph_on", gspace.get("graph_on_choices", [True, False]))
        trial.set_user_attr("graph_on", graph_on)
        if graph_on:
            source = trial.suggest_categorical("source", gspace.get("source_choices", ["snp", "grm"]))
            knn_k = trial.suggest_int("knn_k", *gspace.get("knn_k_range", (1, 7)))
            weighted_edges = trial.suggest_categorical("weighted_edges", gspace.get("weighted_edges_choices", [False, True]))
            symmetrize_mode = trial.suggest_categorical("symmetrize_mode", gspace.get("symmetrize_mode_choices", ["mutual", "union"]))
        else:
            source = "none"
            knn_k = 0
            weighted_edges = False
            symmetrize_mode = "none"
        # laplacian_smoothing ignored by design

        # ---- training hyperparams
        tspace = search_space.get("training", {})
        lr = trial.suggest_float("lr", *tspace.get("lr_loguniform", (1e-4, 5e-3)), log=True)
        weight_decay = trial.suggest_float("weight_decay", *tspace.get("wd_loguniform", (1e-7, 1e-3)), log=True)
        epochs = trial.suggest_int("epochs", *tspace.get("epochs_range", (50, 300)))
        _ = int(tspace.get("patience", 0))  # patience configured but not used in this loop

        # ---- feature selection
        fspace = search_space.get("feature_selection", {})
        use_sel = trial.suggest_categorical("use_snp_selection", fspace.get("use_snp_selection_choices", [False, True]))
        if use_sel:
            ns_min, ns_max = fspace.get("num_snps_range", (5000, 65000))
            step = int(fspace.get("num_snps_step", 5000))
            num_snps = trial.suggest_int("num_snps", ns_min, ns_max, step=step)
        else:
            num_snps = None

        val_scores: List[float] = []

        for fold_idx, (tr_sub, va_sub) in enumerate(inner_splits):
            inner_tr_nodes = outer_train_idx[tr_sub]
            inner_va_nodes = outer_train_idx[va_sub]

            # leakage-safe FS on inner-train
            if use_sel and num_snps is not None:
                feat_idx = top_k_by_abs_corr(
                    X_all[inner_tr_nodes],
                    y_all[inner_tr_nodes],
                    k=min(num_snps, X_all.shape[1]),
                )
                X_use = X_all[:, feat_idx]
            else:
                X_use = X_all

            # Build single global adjacency for this fold
            if graph_on:
                graph_cfg = {
                    "source": source,
                    "knn_k": int(knn_k),
                    "weighted_edges": bool(weighted_edges),
                    "symmetrize_mode": symmetrize_mode,
                    "normalize": True,
                }
                GRM_use = None if GRM_mat is None else GRM_mat
                A = build_global_adjacency(X_use, GRM_use, graph_cfg).tocoo()
            else:
                # Identity adjacency => no neighbors (MLP-like)
                N = X_use.shape[0]
                A = sp.eye(N, dtype=np.float32, format="coo")

            # tensors & masks
            X_t = to_torch(X_use.astype(np.float32), device)
            y_t = to_torch(y_all.astype(np.float32), device)
            A_indices = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long, device=device)
            A_values = torch.tensor(A.data.astype(np.float32), dtype=torch.float32, device=device)
            A_shape = (A.shape[0], A.shape[1])

            train_mask = np.zeros(len(X_use), dtype=bool)
            train_mask[inner_tr_nodes] = True
            val_mask = np.zeros(len(X_use), dtype=bool)
            val_mask[inner_va_nodes] = True

            model = GCN(
                in_dim=X_t.shape[1],
                hidden_dims=hidden_dims,
                dropout=dropout,
                use_bn=use_bn,
            ).to(device)

            # Use a unique reporting window per inner fold to avoid duplicate step warnings
            report_offset = fold_idx * (epochs + 1)

            model = train_one(
                model, X_t, A_indices, A_values, A_shape, y_t,
                train_mask,
                val_mask=val_mask,
                lr=lr, weight_decay=weight_decay,
                epochs=epochs, device=device,
                trial=trial,
                prune_warmup_epochs=int(cfg.get("pruner_warmup_epochs", 5)),
                prune_interval=1,
                report_offset=report_offset,
                enable_pruning=bool(cfg.get("enable_pruning", True)),
            )

            model.eval()
            with torch.no_grad():
                pred = model(X_t, A_indices, A_values, A_shape).detach().cpu().numpy()
            r = pearson_corr(pred[val_mask], y_all[val_mask])
            val_scores.append(r)

        mean_r = float(np.mean(val_scores))
        trial.set_user_attr("mean_inner_r", mean_r)
        return mean_r

    return objective


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config_nested.json")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    base = cfg.get("base_train", {})
    search_space = cfg.get("search_space", {})
    seed = int(base.get("seed", 42))
    set_seed(seed)

    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # paths / columns
    paths = base.get("paths", {})
    target_col = base.get("target_column", "y_adjusted")
    eval_col = base.get("eval_target_column", target_col)
    out_dir = paths.get("output_dir", "outputs/nested_cv")
    out_name = paths.get("output_name", "results")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load with your loader
    X, y, ids, GRM_df = load_data(paths, target_column=target_col, standardize_features=bool(base.get("standardize_features", False)))
    GRM_mat = None if GRM_df is None else GRM_df.values.astype(np.float32, copy=False)

    # Eval target: try NPZ; else fallback to y
    y_eval = try_load_eval_from_npz(paths, eval_col)
    if y_eval is None:
        logging.info(f"Eval target '{eval_col}' not found in NPZ; using training target for eval.")
        y_eval = y.copy()

    # Outer CV
    outer_k = int(cfg.get("outer_folds", 10))
    kf_outer = KFold(n_splits=outer_k, shuffle=True, random_state=seed)

    # Optuna settings
    n_trials = int(cfg.get("n_trials", 50))
    n_startup = int(cfg.get("n_startup_trials", 10))
    pruner_startup = int(cfg.get("pruner_startup_trials", 5))
    pruner_warmup_epochs = int(cfg.get("pruner_warmup_epochs", 5))
    enable_pruning = bool(cfg.get("enable_pruning", True))
    study_name = cfg.get("study_name", "transductive_gcn")
    storage = cfg.get("storage", None)
    show_bar = bool(cfg.get("show_progress_bar", True))
    timeout = cfg.get("timeout_seconds", None)

    results = []
    best_params_per_fold: List[Dict] = []
    fold_id = 0

    for tr_idx, te_idx in kf_outer.split(np.arange(len(X))):
        fold_id += 1
        logging.info(f"[Outer {fold_id}/{outer_k}] Train {len(tr_idx)} | Test {len(te_idx)}")

        # ---- tune on outer-train
        sampler = TPESampler(seed=seed, n_startup_trials=n_startup)
        pruner = (
            MedianPruner(
                n_startup_trials=int(cfg.get("pruner_startup_trials", pruner_startup)),
                n_warmup_steps=int(cfg.get("pruner_warmup_epochs", pruner_warmup_epochs)),
                interval_steps=1,
            )
            if enable_pruning else NopPruner()
        )
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=study_name,
            storage=storage,
            load_if_exists=False,
        )
        obj = objective_factory(X, y, GRM_mat, tr_idx, cfg, search_space, device)
        study.optimize(obj, n_trials=n_trials, timeout=timeout, show_progress_bar=show_bar)

        best = study.best_params
        logging.info(f"Best params (outer fold {fold_id}): {best} | mean inner r = {study.best_value:.4f}")
        best_params_per_fold.append({
            "fold": fold_id,
            "best_params": best,
            "mean_inner_r": float(study.best_value),
        })

        # ---- FS redone on full outer-train
        use_sel_final = best.get("use_snp_selection", False)
        num_snps_final = best.get("num_snps", None) if use_sel_final else None

        if use_sel_final and num_snps_final is not None:
            feat_idx_final = top_k_by_abs_corr(
                X[tr_idx],
                y[tr_idx],
                k=min(int(num_snps_final), X.shape[1]),
            )
            X_use = X[:, feat_idx_final]
        else:
            X_use = X

        # Global graph (transductive) for final fit
        graph_on_best = bool(best.get("graph_on", True))
        if graph_on_best:
            graph_cfg = {
                "source": best.get("source", "snp"),
                "knn_k": int(best.get("knn_k", 3)),
                "weighted_edges": bool(best.get("weighted_edges", False)),
                "symmetrize_mode": best.get("symmetrize_mode", "mutual"),
                "normalize": True,
            }
            A = build_global_adjacency(X_use, GRM_mat, graph_cfg).tocoo()
        else:
            N = X_use.shape[0]
            A = sp.eye(N, dtype=np.float32, format="coo")

        # tensors & masks
        X_t = to_torch(X_use.astype(np.float32), device)
        y_t = to_torch(y.astype(np.float32), device)
        A_indices = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long, device=device)
        A_values = torch.tensor(A.data.astype(np.float32), dtype=torch.float32, device=device)
        A_shape = (A.shape[0], A.shape[1])

        N = len(X_use)
        train_mask = np.zeros(N, dtype=bool)
        train_mask[tr_idx] = True
        test_mask = np.zeros(N, dtype=bool)
        test_mask[te_idx] = True

        # model
        model = GCN(
            in_dim=X_t.shape[1],
            hidden_dims=decode_choice(best["hidden_dims"]),
            dropout=float(best["dropout"]),
            use_bn=bool(best["batch_norm"]),
        ).to(device)

        # final training with tuned epochs/lr/wd
        train_full(
            model, X_t, A_indices, A_values, A_shape, y_t,
            train_mask,
            lr=float(best["lr"]),
            weight_decay=float(best["weight_decay"]),
            max_epochs=int(best["epochs"]),
            device=device,
        )

        # evaluate on outer-test versus y_eval
        model.eval()
        with torch.no_grad():
            pred = model(X_t, A_indices, A_values, A_shape).detach().cpu().numpy()
        r_test = pearson_corr(pred[test_mask], y_eval[test_mask])
        results.append(float(r_test))
        logging.info(f"[Outer {fold_id}] Test Pearson r = {r_test:.4f}")

    logging.info(f"Outer-fold correlations: {results}")
    logging.info(f"Mean r = {np.mean(results):.4f} | Std = {np.std(results):.4f}")

    # save summary
    out_path = os.path.join(out_dir, f"{out_name}.json")
    with open(out_path, "w") as f:
        json.dump({
            "fold_corr": results,
            "mean": float(np.mean(results)),
            "std": float(np.std(results)),
            "best_params_per_fold": best_params_per_fold,
        }, f, indent=2)
    logging.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
