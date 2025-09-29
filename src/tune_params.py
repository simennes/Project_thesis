from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np
import optuna
import torch
import torch.nn as nn

from src.data import load_data
from src.graph import build_global_adjacency, partition_train_graph
from src.gcn import GCN
from src.utils import (set_seed,
                       to_torch_sparse,
                       _optimizer,
                       _select_top_snps_by_abs_corr, _pearson_corr, _split_indices, encode_choices_for_optuna, decode_choice)

def _train_single(
    model: GCN,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    A_idx, A_val, A_shape,
    X_va: np.ndarray,
    y_va: np.ndarray,
    A_va_idx, A_va_val, A_va_shape,
    optimizer,
    epochs: int,
    patience: int,
    device: torch.device,
    loss_fn: nn.Module,
    trial: optuna.trial.Trial | None = None,
    report_offset: int = 0,   # <-- unique offset to avoid duplicate step warnings
) -> float:
    """Train on train graph, validate on val graph; return BEST validation Pearson r (maximize)."""
    Xtr_t = torch.from_numpy(X_tr).to(device)
    ytr_t = torch.from_numpy(y_tr).to(device)
    Xva_t = torch.from_numpy(X_va).to(device)

    best_r = -1.0
    best_state = None
    no_imp = 0

    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred_tr = model(Xtr_t, A_idx, A_val, A_shape)
        loss = loss_fn(pred_tr, ytr_t)
        loss.backward()
        optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            pred_va = model(Xva_t, A_va_idx, A_va_val, A_va_shape).cpu().numpy()
        r = _pearson_corr(y_va, pred_va)

        # unique step number (offset + ep) to silence Optuna warnings
        if trial is not None:
            trial.report(r, report_offset + ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if r > best_r + 1e-12:
            best_r = r
            no_imp = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if patience > 0 and no_imp >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_r


# -----------------------------
# Optuna bits
# -----------------------------
def _sample_hparams(trial: optuna.trial.Trial, base_cfg: Dict[str, Any], search: Dict[str, Any]) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(base_cfg))  # deep copy

    mspace = search.get("model", {})
    gspace = search.get("graph", {})
    tspace = search.get("training", {})
    fspace = search.get("feature_selection", {})

    # --- Model ---
    # Avoid Optuna warning by sampling strings and decoding to list
    hd_choices_cfg = mspace.get("hidden_dims_choices", [[128, 64], [256, 128], [64, 64]])
    hd_choices_str = encode_choices_for_optuna(hd_choices_cfg)
    hd_pick_str = trial.suggest_categorical("hidden_dims", hd_choices_str)
    cfg["model"]["hidden_dims"] = decode_choice(hd_pick_str)

    dr_min, dr_max = mspace.get("dropout_range", (0.0, 0.6))
    cfg["model"]["dropout"] = trial.suggest_float("dropout", dr_min, dr_max)
    cfg["model"]["batch_norm"] = trial.suggest_categorical("batch_norm", [True, False])

    # --- Graph (source comes from base_cfg["graph"]["source"]) ---
    cfg["graph"]["knn_k"] = trial.suggest_int("knn_k", *gspace.get("knn_k_range", (3, 10)))
    cfg["graph"]["weighted_edges"] = trial.suggest_categorical("weighted_edges", [False, True])
    cfg["graph"]["symmetrize_mode"] = trial.suggest_categorical(
        "symmetrize_mode", gspace.get("symmetrize_mode_choices", ["mutual", "union"])
    )
    cfg["graph"]["laplacian_smoothing"] = trial.suggest_categorical(
        "laplacian_smoothing", gspace.get("laplacian_smoothing_choices", [True, False])
    )
    cfg["graph"]["ensemble_models"] = trial.suggest_int("ensemble_models", *gspace.get("ensemble_models_range", (1, 8)))

    # --- Training ---
    cfg["training"]["lr"] = trial.suggest_float("lr", *tspace.get("lr_loguniform", (1e-4, 5e-3)), log=True)
    cfg["training"]["weight_decay"] = trial.suggest_float("weight_decay", *tspace.get("wd_loguniform", (1e-7, 1e-3)), log=True)
    cfg["training"]["optimizer"] = trial.suggest_categorical("optimizer", tspace.get("optimizer_choices", ["adam", "sgd"]))
    cfg["training"]["epochs"] = trial.suggest_int("epochs", *tspace.get("epochs_range", (50, 300)), step=10)
    cfg["training"]["patience"] = tspace.get("patience", 0)

    # fixed fractions
    cfg["graph"]["val_fraction"] = base_cfg["graph"].get("val_fraction", 0.1)
    cfg["graph"]["test_fraction"] = base_cfg["graph"].get("test_fraction", 0.1)

    # --- Feature selection (conditional) ---
    use_sel = trial.suggest_categorical("use_snp_selection", fspace.get("use_snp_selection_choices", [False, True]))
    cfg.setdefault("feature_selection", {})
    cfg["feature_selection"]["use_snp_selection"] = bool(use_sel)
    if use_sel:
        ns_min, ns_max = fspace.get("num_snps_range", (5_000, 65_000))
        step = int(fspace.get("num_snps_step", 5_000))
        cfg["feature_selection"]["num_snps"] = trial.suggest_int("num_snps", ns_min, ns_max, step=step)
    else:
        # Do NOT sample num_snps when selection is off (prevents TPE noise)
        cfg["feature_selection"]["num_snps"] = None

    return cfg


def objective(trial: optuna.trial.Trial, base_cfg: Dict[str, Any], search: Dict[str, Any]) -> float:
    """Return validation Pearson correlation (maximize)."""
    cfg = _sample_hparams(trial, base_cfg, search)

    seed = base_cfg.get("seed", 42)
    set_seed(seed)
    torch.set_num_threads(8)
    os.environ["OMP_NUM_THREADS"] = "8"

    # Load data once per trial
    X, y, ids, GRM_df = load_data(
        base_cfg["paths"],
        target_column=base_cfg.get("target_column", "y_adjusted"),
        standardize_features=base_cfg.get("standardize_features", False),
    )

    # Build ONE global adjacency from ALL SNPs (paper-faithful)
    A_global_csr = build_global_adjacency(X, GRM_df, {**base_cfg["graph"], **cfg["graph"]})

    # Split nodes; induce train/val CSR from the SAME global adjacency
    n = X.shape[0]
    tr_idx, va_idx, _ = _split_indices(n, cfg["graph"]["val_fraction"], cfg["graph"]["test_fraction"], seed)
    A_train_csr = A_global_csr[tr_idx][:, tr_idx].tocsr()
    A_val_csr = A_global_csr[va_idx][:, va_idx].tocsr()

    # Feature selection columns (computed on TRAIN ONLY, applied to train/val)
    if cfg.get("feature_selection", {}).get("use_snp_selection", False):
        p = X.shape[1]
        k = int(min(cfg["feature_selection"]["num_snps"], p))
        cols_sel = _select_top_snps_by_abs_corr(X[tr_idx], y[tr_idx], k)
    else:
        cols_sel = slice(None)

    # Torch sparse
    A_tr_idx, A_tr_val, A_tr_shape = to_torch_sparse(A_train_csr)
    A_va_idx, A_va_val, A_va_shape = to_torch_sparse(A_val_csr)
    if torch.cuda.is_available():
        ngpus = torch.cuda.device_count()
        device_id = trial.number % ngpus
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")

    A_tr_idx, A_tr_val = A_tr_idx.to(device), A_tr_val.to(device)
    A_va_idx, A_va_val = A_va_idx.to(device), A_va_val.to(device)

    loss_fn = nn.MSELoss()
    ensemble = int(cfg["graph"]["ensemble_models"])

    # SINGLE model
    if ensemble <= 1:
        X_tr = X[tr_idx][:, cols_sel]
        X_va = X[va_idx][:, cols_sel]
        model = GCN(
            in_dim=X_tr.shape[1],
            hidden_dims=cfg["model"]["hidden_dims"],
            dropout=cfg["model"]["dropout"],
            use_bn=cfg["model"]["batch_norm"],
        ).to(device)
        opt = _optimizer(cfg["training"]["optimizer"], model.parameters(), cfg["training"]["lr"], cfg["training"]["weight_decay"])
        best_r = _train_single(
            model, X_tr, y[tr_idx], A_tr_idx, A_tr_val, A_tr_shape,
            X_va, y[va_idx], A_va_idx, A_va_val, A_va_shape,
            opt, cfg["training"]["epochs"], cfg["training"]["patience"], device, loss_fn, trial,
            report_offset=0,
        )
        return float(best_r)

    # ENSEMBLE: partition training graph into disjoint connected subgraphs
    parts = partition_train_graph(A_train_csr, ensemble)
    submodels: List[GCN] = []
    ep = int(cfg["training"]["epochs"])
    for i, nodes in enumerate(parts):
        if not nodes:
            continue
        A_sub = A_train_csr[nodes][:, nodes].tocsr()
        A_sub_idx, A_sub_val, A_sub_shape = to_torch_sparse(A_sub)
        A_sub_idx, A_sub_val = A_sub_idx.to(device), A_sub_val.to(device)

        X_sub = X[tr_idx][nodes][:, cols_sel]
        y_sub = y[tr_idx][nodes]

        model_i = GCN(
            in_dim=X_sub.shape[1],
            hidden_dims=cfg["model"]["hidden_dims"],
            dropout=cfg["model"]["dropout"],
            use_bn=cfg["model"]["batch_norm"],
        ).to(device)
        opt_i = _optimizer(cfg["training"]["optimizer"], model_i.parameters(), cfg["training"]["lr"], cfg["training"]["weight_decay"])

        # Give each submodel a unique reporting window to avoid duplicate step numbers
        _ = _train_single(
            model_i,
            X_sub, y_sub,
            A_sub_idx, A_sub_val, A_sub_shape,
            X[va_idx][:, cols_sel], y[va_idx],
            A_va_idx, A_va_val, A_va_shape,
            opt_i, ep, cfg["training"]["patience"], device, nn.MSELoss(), trial,
            report_offset=i * (ep + 1),
        )
        submodels.append(model_i)

    # Validation ensemble = MEAN of submodels (paper’s Algorithm 2)
    with torch.no_grad():
        X_val_t = torch.from_numpy(X[va_idx][:, cols_sel]).to(device)
        preds = [m(X_val_t, A_va_idx, A_va_val, A_va_shape) for m in submodels]
        yhat_val = torch.stack(preds, dim=1).mean(dim=1).cpu().numpy()
    return float(_pearson_corr(y[va_idx], yhat_val))


def main(cfg_path: str) -> None:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    base_cfg = cfg["base_train"]
    search = cfg.get("search_space", {})

    sampler = optuna.samplers.TPESampler(
        seed=base_cfg.get("seed", 42),
        n_startup_trials=cfg.get("n_startup_trials", 10),
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=cfg.get("pruner_startup_trials", 5),
        n_warmup_steps=cfg.get("pruner_warmup_epochs", 5),
    )

    storage = cfg.get("storage", None)  # e.g. "sqlite:///optuna.db"
    study_name = cfg.get("study_name", "gcnrs_tuning")

    # IMPORTANT: maximize Pearson correlation
    if storage:
        study = optuna.create_study(direction="maximize", study_name=study_name, storage=storage, load_if_exists=True, sampler=sampler, pruner=pruner)
    else:
        study = optuna.create_study(direction="maximize", study_name=study_name, sampler=sampler, pruner=pruner)

    study.optimize(
        lambda tr: objective(tr, base_cfg, search),
        n_trials=cfg.get("n_trials", 40),
        timeout=cfg.get("timeout_seconds", None),
        n_jobs=int(cfg.get("n_jobs", 1)),
        gc_after_trial=True,
        show_progress_bar=bool(cfg.get("show_progress_bar", False)),
    )

    outdir = base_cfg["paths"]["output_dir"]
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "tuning_results.json"), "w", encoding="utf-8") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, f, indent=2)

    print("Best Pearson r:", study.best_value)
    print("Best params:", study.best_params)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Optuna hyperparameter tuning for GCN-RS (Algorithm 2) with optional SNP selection")
    ap.add_argument("--config", type=str, required=True, help="Path to config_tune.json")
    args = ap.parse_args()
    main(args.config)
