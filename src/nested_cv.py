from __future__ import annotations
import argparse
import json
import os
import logging
from typing import Any, Dict, List

import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

from src.data import load_data
from src.graph import build_global_adjacency, partition_train_graph
from src.gcn import GCN
from src.utils import (set_seed, to_sparse, save_json,
                       _optimizer, _select_top_snps_by_abs_corr, _pearson_corr,
                       encode_choices_for_optuna, decode_choice)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def _merge_best_params(cfg: Dict[str, Any], best_params: Dict[str, Any]) -> Dict[str, Any]:
    """Overlay Optuna best_params into base config (graph/model/training keys)."""
    out = json.loads(json.dumps(cfg))  # deep copy of base config
    # Graph params
    g = out.setdefault("graph", {})
    for key in ["knn_k", "weighted_edges", "symmetrize_mode", "laplacian_smoothing", "ensemble_models"]:
        if key in best_params:
            g[key] = best_params[key]
    # Model params
    m = out.setdefault("model", {})
    if "hidden_dims" in best_params:
        # If hidden_dims came out as a JSON string, decode it
        if isinstance(best_params["hidden_dims"], str):
            try:
                m["hidden_dims"] = json.loads(best_params["hidden_dims"])
            except json.JSONDecodeError:
                raise ValueError("hidden_dims in best_params is a string but not valid JSON.")
        else:
            m["hidden_dims"] = best_params["hidden_dims"]
    for key in ["dropout", "batch_norm"]:
        if key in best_params:
            m[key] = best_params[key]
    # Training params
    t = out.setdefault("training", {})
    for key in ["lr", "weight_decay", "optimizer", "epochs", "patience", "loss"]:
        if key in best_params:
            t[key] = best_params[key]
    # Feature selection params
    fs = out.setdefault("feature_selection", {})
    if "use_snp_selection" in best_params:
        fs["use_snp_selection"] = bool(best_params["use_snp_selection"])
    if "num_snps" in best_params and best_params.get("use_snp_selection", False):
        fs["num_snps"] = int(best_params["num_snps"])
    return out

def main(config_path: str) -> None:
    # Load nested CV configuration
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    base_cfg = cfg["base_train"]
    search_space = cfg.get("search_space", {})
    outer_folds = int(cfg.get("outer_folds", 10))   # default outer CV = 10
    inner_folds = int(cfg.get("inner_folds", 0))    # if >1, use inner K-fold; if 0 or 1, use val fraction
    inner_val_frac = float(cfg.get("inner_val_fraction", base_cfg.get("graph", {}).get("val_fraction", 0.1)))
    # Determine inner validation strategy
    use_inner_kfold = inner_folds > 1
    if not use_inner_kfold:
        # If inner_folds not specified or <=1, use a hold-out val fraction
        inner_folds = None
        # Ensure fraction is in (0,1)
        if not (0.0 < inner_val_frac < 1.0):
            inner_val_frac = 0.1

    # Set random seed for reproducibility
    seed = base_cfg.get("seed", 42)
    set_seed(seed)

    # Load entire dataset once
    logging.info("Loading data...")
    X, y, ids, GRM_df = load_data(
        base_cfg["paths"],
        target_column=base_cfg.get("target_column", "y_adjusted"),
        standardize_features=base_cfg.get("standardize_features", False),
    )
    n = X.shape[0]
    ids = np.asarray(ids) if ids is not None else np.arange(n)

    # Prepare outer CV splitter
    kf_outer = KFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    # Set up arrays to collect out-of-fold predictions and fold indices
    oof_pred = np.zeros(n, dtype=np.float32)
    oof_fold = -np.ones(n, dtype=int)
    fold_metrics: List[Dict[str, float]] = []
    best_params_list: List[Dict[str, Any]] = []

    # Loop over outer folds
    for fold_idx, (train_idx, test_idx) in enumerate(kf_outer.split(np.arange(n)), start=1):
        logging.info(f"Outer fold {fold_idx}/{outer_folds}: training on {len(train_idx)} samples, testing on {len(test_idx)} samples")
        # Subset data for this fold
        X_train_full = X[train_idx]
        y_train_full = y[train_idx]
        if GRM_df is not None:
            # Subset the GRM matrix to training individuals (iloc preserves index order of train_idx list)
            GRM_train = GRM_df.iloc[train_idx, train_idx]
        else:
            GRM_train = None

        # Define the objective function for inner hyperparameter tuning on this fold
        def objective(trial: optuna.trial.Trial) -> float:
            # Sample hyperparameters using Optuna (based on provided search_space ranges/choices)
            # Start with a deep copy of base config to fill in
            cfg_trial = json.loads(json.dumps(base_cfg))
            # --- Model hyperparams ---
            mspace = search_space.get("model", {})
            hd_choices = mspace.get("hidden_dims_choices", [[128, 64], [256, 128], [64, 64]])
            # Use JSON encoding for list choices to avoid Optuna warnings
            hd_choices_str = encode_choices_for_optuna(hd_choices)
            hd_choice = trial.suggest_categorical("hidden_dims", hd_choices_str)
            cfg_trial["model"]["hidden_dims"] = decode_choice(hd_choice)
            cfg_trial["model"]["dropout"] = trial.suggest_float("dropout", *mspace.get("dropout_range", (0.0, 0.6)))
            cfg_trial["model"]["batch_norm"] = trial.suggest_categorical("batch_norm", [True, False])
            # --- Graph hyperparams ---
            gspace = search_space.get("graph", {})
            cfg_trial.setdefault("graph", {})
            cfg_trial["graph"]["knn_k"] = trial.suggest_int("knn_k", *gspace.get("knn_k_range", (3, 10)))
            cfg_trial["graph"]["weighted_edges"] = trial.suggest_categorical("weighted_edges", [False, True])
            cfg_trial["graph"]["symmetrize_mode"] = trial.suggest_categorical(
                "symmetrize_mode", gspace.get("symmetrize_mode_choices", ["mutual", "union"])
            )
            cfg_trial["graph"]["laplacian_smoothing"] = trial.suggest_categorical(
                "laplacian_smoothing", gspace.get("laplacian_smoothing_choices", [True, False])
            )
            cfg_trial["graph"]["ensemble_models"] = trial.suggest_int("ensemble_models", *gspace.get("ensemble_models_range", (1, 8)))
            # Ensure we carry over any fixed fractions from base config (if present)
            cfg_trial["graph"]["val_fraction"] = base_cfg.get("graph", {}).get("val_fraction", 0.1)
            cfg_trial["graph"]["test_fraction"] = base_cfg.get("graph", {}).get("test_fraction", 0.1)
            # --- Training hyperparams ---
            tspace = search_space.get("training", {})
            cfg_trial.setdefault("training", {})
            cfg_trial["training"]["lr"] = trial.suggest_float("lr", *tspace.get("lr_loguniform", (1e-4, 5e-3)), log=True)
            cfg_trial["training"]["weight_decay"] = trial.suggest_float("weight_decay", *tspace.get("wd_loguniform", (1e-7, 1e-3)), log=True)
            cfg_trial["training"]["optimizer"] = trial.suggest_categorical("optimizer", tspace.get("optimizer_choices", ["adam", "sgd"]))
            cfg_trial["training"]["epochs"] = trial.suggest_int("epochs", *tspace.get("epochs_range", (50, 300)), step=10)
            cfg_trial["training"]["patience"] = tspace.get("patience", 0)  # patience not tuned, just use base or default
            # Loss function (MAE or MSE)
            cfg_trial["training"]["loss"] = trial.suggest_categorical(
                "loss", tspace.get("loss_choices", ["mse"]))

            # --- Feature selection hyperparams ---
            fspace = search_space.get("feature_selection", {})
            use_sel = trial.suggest_categorical("use_snp_selection", fspace.get("use_snp_selection_choices", [False, True]))
            cfg_trial.setdefault("feature_selection", {})
            cfg_trial["feature_selection"]["use_snp_selection"] = bool(use_sel)
            if cfg_trial["feature_selection"]["use_snp_selection"]:
                ns_min, ns_max = fspace.get("num_snps_range", (5000, 65000))
                step = int(fspace.get("num_snps_step", 5000))
                cfg_trial["feature_selection"]["num_snps"] = trial.suggest_int("num_snps", ns_min, ns_max, step=step)
            else:
                cfg_trial["feature_selection"]["num_snps"] = None

            # Set random seed for this trial (use base seed for reproducibility)
            set_seed(base_cfg.get("seed", 42))
            torch.set_num_threads(8)  # limit CPU threads for consistency

            # Build one global adjacency on the **outer training data** for this hyperparam trial
            A_global_train = build_global_adjacency(X_train_full, GRM_train, cfg_trial["graph"])
            # Prepare inner CV splits on the training set
            if use_inner_kfold:
                inner_kf = KFold(n_splits=inner_folds, shuffle=True, random_state=seed + fold_idx)
                inner_splits = list(inner_kf.split(np.arange(len(train_idx))))  # splits on indices [0 .. len(train_idx)-1]
            else:
                # Single hold-out split using inner_val_frac
                train_loc_idx, val_loc_idx, _ = _select_train_val_indices(len(train_idx), inner_val_frac, seed + fold_idx)
                inner_splits = [ (train_loc_idx, val_loc_idx) ]
            # Evaluate this hyperparameter set via inner CV
            fold_r_values: List[float] = []
            for inner_train_loc, inner_val_loc in inner_splits:
                # Induce subgraph for inner train and inner val from the global train adjacency
                A_tr = A_global_train[inner_train_loc][:, inner_train_loc].tocsr()
                A_val = A_global_train[inner_val_loc][:, inner_val_loc].tocsr()
                # Perform feature selection on inner train if enabled
                if cfg_trial["feature_selection"].get("use_snp_selection", False):
                    k_snps = int(min(cfg_trial["feature_selection"]["num_snps"] or X_train_full.shape[1], X_train_full.shape[1]))
                    cols_sel = _select_top_snps_by_abs_corr(X_train_full[inner_train_loc], y_train_full[inner_train_loc], k_snps)
                else:
                    cols_sel = slice(None)
                # Convert adjacency matrices to torch sparse tensors for model input
                # Default to CPU; if CUDA is available, distribute trials across GPUs
                device = torch.device("cpu")
                if torch.cuda.is_available():
                    ngpu = torch.cuda.device_count()
                    device_id = trial.number % max(1, ngpu)
                    device = torch.device(f"cuda:{device_id}")
                A_tr_idx, A_tr_val, A_tr_shape = to_sparse(A_tr, device)
                A_val_idx, A_val_val, A_val_shape = to_sparse(A_val, device)
                # Initialize model (single GCN or ensemble handled inside training loop)
                ensemble_count = int(cfg_trial["graph"].get("ensemble_models", 1))
                loss_name = cfg_trial["training"].get("loss", "mse").lower()
                loss_fn = nn.L1Loss() if loss_name == "mae" else nn.MSELoss()
                # Training on inner training set:
                if ensemble_count <= 1:
                    # Single model training on inner_train_loc
                    model = GCN(
                        in_dim=X_train_full[inner_train_loc][:, cols_sel].shape[1],
                        hidden_dims=cfg_trial["model"]["hidden_dims"],
                        dropout=cfg_trial["model"]["dropout"],
                        use_bn=cfg_trial["model"]["batch_norm"]
                    ).to(device)
                    optimizer = _optimizer(cfg_trial["training"]["optimizer"], model.parameters(),
                                            cfg_trial["training"]["lr"], cfg_trial["training"]["weight_decay"])
                    # Prepare torch tensors for features/labels
                    Xtr_t = torch.from_numpy(X_train_full[inner_train_loc][:, cols_sel]).to(device)
                    ytr_t = torch.from_numpy(y_train_full[inner_train_loc]).to(device)
                    Xval_t = torch.from_numpy(X_train_full[inner_val_loc][:, cols_sel]).to(device)
                    best_r = -1.0
                    no_imp = 0
                    # Training loop with early stopping on inner validation
                    for ep in range(1, int(cfg_trial["training"]["epochs"]) + 1):
                        model.train()
                        optimizer.zero_grad()
                        pred_tr = model(Xtr_t, A_tr_idx, A_tr_val, A_tr_shape)
                        loss = loss_fn(pred_tr, ytr_t)
                        loss.backward()
                        optimizer.step()
                        model.eval()
                        with torch.no_grad():
                            pred_val = model(Xval_t, A_val_idx, A_val_val, A_val_shape).cpu().numpy()
                        r_val = _pearson_corr(y_train_full[inner_val_loc], pred_val)
                        if r_val > best_r + 1e-12:
                            best_r = r_val
                            no_imp = 0
                        else:
                            no_imp += 1
                            if cfg_trial["training"]["patience"] > 0 and no_imp >= cfg_trial["training"]["patience"]:
                                break  # early stopping
                    fold_r_values.append(best_r)
                else:
                    # Ensemble model training: partition inner_train graph and train submodels
                    A_train_csr = A_tr  # adjacency for inner train nodes
                    parts = partition_train_graph(A_train_csr, ensemble_count)
                    # Train each submodel on its partition
                    submodels: List[GCN] = []
                    epochs = int(cfg_trial["training"]["epochs"])
                    loss_name = cfg_trial["training"].get("loss", "mse").lower()
                    loss_fn = nn.L1Loss() if loss_name == "mae" else nn.MSELoss()
                    for i, nodes in enumerate(parts):
                        if not nodes:  # skip empty partition (if any)
                            continue
                        # Build subgraph for this partition
                        A_sub = A_train_csr[nodes][:, nodes].tocsr()
                        A_sub_idx, A_sub_val, A_sub_shape = to_sparse(A_sub, device)
                        X_sub = X_train_full[inner_train_loc][nodes][:, cols_sel]
                        y_sub = y_train_full[inner_train_loc][nodes]
                        # Initialize submodel and optimizer
                        model_i = GCN(
                            in_dim=X_sub.shape[1],
                            hidden_dims=cfg_trial["model"]["hidden_dims"],
                            dropout=cfg_trial["model"]["dropout"],
                            use_bn=cfg_trial["model"]["batch_norm"]
                        ).to(device)
                        opt_i = _optimizer(cfg_trial["training"]["optimizer"], model_i.parameters(),
                                           cfg_trial["training"]["lr"], cfg_trial["training"]["weight_decay"])
                        # Train submodel with early stopping on same inner val
                        best_r_i = -1.0
                        no_imp_i = 0
                        Xsub_t = torch.from_numpy(X_sub).to(device)
                        ysub_t = torch.from_numpy(y_sub).to(device)
                        Xval_t = torch.from_numpy(X_train_full[inner_val_loc][:, cols_sel]).to(device)
                        for ep in range(1, epochs + 1):
                            model_i.train()
                            opt_i.zero_grad()
                            pred_sub = model_i(Xsub_t, A_sub_idx, A_sub_val, A_sub_shape)
                            loss = loss_fn(pred_sub, ysub_t)
                            loss.backward()
                            opt_i.step()
                            model_i.eval()
                            with torch.no_grad():
                                pred_val = model_i(Xval_t, A_val_idx, A_val_val, A_val_shape).cpu().numpy()
                            r_val = _pearson_corr(y_train_full[inner_val_loc], pred_val)
                            if r_val > best_r_i + 1e-12:
                                best_r_i = r_val
                                no_imp_i = 0
                            else:
                                no_imp_i += 1
                                if cfg_trial["training"]["patience"] > 0 and no_imp_i >= cfg_trial["training"]["patience"]:
                                    break
                        submodels.append(model_i)
                    # After training all submodels, evaluate ensemble on inner val
                    with torch.no_grad():
                        Xval_t = torch.from_numpy(X_train_full[inner_val_loc][:, cols_sel]).to(device)
                        preds = [m(Xval_t, A_val_idx, A_val_val, A_val_shape) for m in submodels]
                        preds_stack = torch.stack(preds, dim=1).mean(dim=1).cpu().numpy()
                    r_ens = _pearson_corr(y_train_full[inner_val_loc], preds_stack)
                    fold_r_values.append(r_ens)
            # Compute average inner validation Pearson r across inner folds
            mean_r = float(np.mean(fold_r_values))
            # Optionally prune trial if needed (Optuna handles via trial.report in training loop above)
            return mean_r

        # Set up Optuna study for this fold's hyperparameter tuning
        sampler = optuna.samplers.TPESampler(seed=base_cfg.get("seed", 42),
                                             n_startup_trials=cfg.get("n_startup_trials", 10))
        pruner = optuna.pruners.MedianPruner(n_startup_trials=cfg.get("pruner_startup_trials", 5),
                                             n_warmup_steps=cfg.get("pruner_warmup_epochs", 5))
        study_name = cfg.get("study_name", f"nestedcv_fold{fold_idx}")
        storage = cfg.get("storage", None)
        if storage:
            study = optuna.create_study(direction="maximize", study_name=study_name,
                                        storage=storage, load_if_exists=True,
                                        sampler=sampler, pruner=pruner)
        else:
            study = optuna.create_study(direction="maximize", study_name=study_name,
                                        sampler=sampler, pruner=pruner)
        logging.info(f"Starting hyperparameter tuning for fold {fold_idx} (n_trials={cfg.get('n_trials', 40)})...")
        study.optimize(objective,
                       n_trials=cfg.get("n_trials", 40),
                       timeout=cfg.get("timeout_seconds", None),
                       n_jobs=int(cfg.get("n_jobs", 1)),
                       gc_after_trial=True,
                       show_progress_bar=bool(cfg.get("show_progress_bar", False)))
        # Retrieve best hyperparameters for this fold
        best_params = study.best_params
        logging.info(f"Fold {fold_idx}: Best inner validation Pearson r = {study.best_value:.4f}")
        logging.info(f"Fold {fold_idx}: Best hyperparameters = {best_params}")
        best_params_list.append({"fold": fold_idx, "best_params": best_params})

        # Merge best params into base config for final model training
        cfg_fold = _merge_best_params(base_cfg, best_params)
    # Build one global adjacency on ALL data (train+test) using the best graph params
        A_global_all = build_global_adjacency(X, GRM_df, cfg_fold["graph"])
        # Induce train and test subgraph for this fold
        A_train = A_global_all[train_idx][:, train_idx].tocsr()
        A_test = A_global_all[test_idx][:, test_idx].tocsr()
        # If feature selection was used, select top SNPs on *full outer train* and reduce features
        if cfg_fold.get("feature_selection", {}).get("use_snp_selection", False):
            k = int(min(cfg_fold["feature_selection"].get("num_snps", X.shape[1]), X.shape[1]))
            cols_sel = _select_top_snps_by_abs_corr(X_train_full, y_train_full, k)
        else:
            cols_sel = slice(None)
        # Convert adjacency to torch sparse
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        A_tr_idx, A_tr_val, A_tr_shape = to_sparse(A_train, device)
        A_te_idx, A_te_val, A_te_shape = to_sparse(A_test, device)
        # Train final model on all outer training data with best hyperparams (no inner val, train to full epochs)
        ensemble_count = int(cfg_fold["graph"].get("ensemble_models", 1))
        yhat_test: np.ndarray
        loss_name_final = cfg_fold.get("training", {}).get("loss", "mse").lower()
        loss_fn_final = nn.L1Loss() if loss_name_final == "mae" else nn.MSELoss()
        if ensemble_count <= 1:
            # Train a single GCN model on full outer train
            model = GCN(
                in_dim=X_train_full[:, cols_sel].shape[1],
                hidden_dims=cfg_fold["model"]["hidden_dims"],
                dropout=cfg_fold["model"]["dropout"],
                use_bn=cfg_fold["model"]["batch_norm"]
            ).to(device)
            optimizer = _optimizer(cfg_fold["training"]["optimizer"], model.parameters(),
                                   cfg_fold["training"]["lr"], cfg_fold["training"].get("weight_decay", 0.0))
            X_tr_t = torch.from_numpy(X_train_full[:, cols_sel]).to(device)
            y_tr_t = torch.from_numpy(y_train_full).to(device)
            # Train for the specified number of epochs (using all training data, no early stopping)
            epochs = int(cfg_fold["training"]["epochs"])
            for ep in range(1, epochs + 1):
                model.train()
                optimizer.zero_grad()
                pred = model(X_tr_t, A_tr_idx, A_tr_val, A_tr_shape)
                loss = loss_fn_final(pred, y_tr_t)
                loss.backward()
                optimizer.step()
            # After training, predict on the test fold
            model.eval()
            with torch.no_grad():
                X_te_t = torch.from_numpy(X[test_idx][:, cols_sel]).to(device)
                yhat_test = model(X_te_t, A_te_idx, A_te_val, A_te_shape).cpu().numpy()
        else:
            # Train an ensemble of GCN models on full outer train (disjoint subgraphs)
            A_train_csr = A_train  # adjacency of training nodes
            parts = partition_train_graph(A_train_csr, ensemble_count)
            submodels: List[GCN] = []
            epochs = int(cfg_fold["training"]["epochs"])
            for i, nodes in enumerate(parts):
                if not nodes:
                    continue
                # Partition subgraph
                A_sub = A_train_csr[nodes][:, nodes].tocsr()
                A_sub_idx, A_sub_val, A_sub_shape = to_sparse(A_sub, device)
                X_sub = X_train_full[nodes][:, cols_sel]
                y_sub = y_train_full[nodes]
                # Init submodel
                model_i = GCN(
                    in_dim=X_sub.shape[1],
                    hidden_dims=cfg_fold["model"]["hidden_dims"],
                    dropout=cfg_fold["model"]["dropout"],
                    use_bn=cfg_fold["model"]["batch_norm"]
                ).to(device)
                opt_i = _optimizer(cfg_fold["training"]["optimizer"], model_i.parameters(),
                                   cfg_fold["training"]["lr"], cfg_fold["training"].get("weight_decay", 0.0))
                Xsub_t = torch.from_numpy(X_sub).to(device)
                ysub_t = torch.from_numpy(y_sub).to(device)
                # Train submodel for fixed epochs (no early stop)
                for ep in range(1, epochs + 1):
                    model_i.train()
                    opt_i.zero_grad()
                    pred_sub = model_i(Xsub_t, A_sub_idx, A_sub_val, A_sub_shape)
                    loss = loss_fn_final(pred_sub, ysub_t)
                    loss.backward()
                    opt_i.step()
                submodels.append(model_i)
            # Ensemble predict on test fold
            with torch.no_grad():
                X_te_t = torch.from_numpy(X[test_idx][:, cols_sel]).to(device)
                preds = [m(X_te_t, A_te_idx, A_te_val, A_te_shape) for m in submodels]
                yhat_test = torch.stack(preds, dim=1).mean(dim=1).cpu().numpy()

        # Store predictions and compute metric for this fold
        oof_pred[test_idx] = yhat_test.flatten()
        oof_fold[test_idx] = fold_idx
        r_test = _pearson_corr(y[test_idx], yhat_test)
        fold_metrics.append({"fold": fold_idx, "pearson_r": float(r_test)})
        logging.info(f"Fold {fold_idx} | Test Pearson r = {r_test:.4f}")

    # Compute overall Pearson correlation on all out-of-fold predictions
    overall_r = _pearson_corr(y, oof_pred)
    logging.info(f"Overall Pearson r (nested CV, {outer_folds}-fold) = {overall_r:.4f}")

    # Save outputs: predictions and metrics
    out_dir = base_cfg["paths"].get("output_dir", ".")
    os.makedirs(out_dir, exist_ok=True)
    # Save OOF predictions for all samples
    import pandas as pd
    pd.DataFrame({
        "id": ids,
        "fold": oof_fold,
        "y_true": y,
        "y_pred": oof_pred
    }).to_csv(os.path.join(out_dir, "nested_oof_predictions.csv"), index=False)
    # Save metrics and best hyperparams
    results = {
        "overall": {"pearson_r": overall_r},
        "per_fold": fold_metrics,
        "best_params_per_fold": best_params_list
    }
    save_json(results, os.path.join(out_dir, "nested_cv_metrics.json"))
    logging.info(f"Saved nested CV predictions to nested_oof_predictions.csv and metrics to nested_cv_metrics.json in {out_dir}")

# Utility: function to get a single train/val split indices for hold-out
def _select_train_val_indices(n: int, val_fraction: float, seed: int):
    # Reuse utils._split_indices if available or implement similar
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = int(n * val_fraction)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nested cross-validation for GCN/GCN-RS with inner hyperparam tuning")
    parser.add_argument("--config", type=str, required=True, help="Path to nested CV config JSON")
    args = parser.parse_args()
    main(args.config)
