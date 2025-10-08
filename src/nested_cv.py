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
import gc
from sklearn.model_selection import KFold

from src.data import load_data
from src.graph import build_global_adjacency, partition_train_graph
from src.gcn import GCN
from src.utils import (
    set_seed,
    to_sparse,
    save_json,
    _optimizer,
    _select_top_snps_by_abs_corr,
    _pearson_corr,
    encode_choices_for_optuna,
    decode_choice,
)

THREADS_PER_TRIAL = int(os.getenv("THREADS_PER_TRIAL", "4"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS_PER_TRIAL))
os.environ.setdefault("MKL_NUM_THREADS", str(THREADS_PER_TRIAL))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(THREADS_PER_TRIAL))
torch.set_num_threads(THREADS_PER_TRIAL)  # set once before any parallel work
torch.set_num_interop_threads(1)          # set once before any parallel work

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
    for key in ["source", "knn_k", "weighted_edges", "symmetrize_mode", "laplacian_smoothing", "ensemble_models"]:
        if key in best_params:
            g[key] = best_params[key]
    # Model params
    m = out.setdefault("model", {})
    if "hidden_dims" in best_params:
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
    outer_folds = int(cfg.get("outer_folds", 10))
    inner_folds = int(cfg.get("inner_folds", 0))  # if >1, use inner K-fold; else use val fraction
    inner_val_frac = float(cfg.get("inner_val_fraction", base_cfg.get("graph", {}).get("val_fraction", 0.1)))

    # Determine inner validation strategy
    use_inner_kfold = inner_folds > 1
    if not use_inner_kfold:
        inner_folds = None
        if not (0.0 < inner_val_frac < 1.0):
            inner_val_frac = 0.1

    # Seed
    seed = base_cfg.get("seed", 42)
    set_seed(seed)

    # Load data once
    logging.info("Loading data...")
    X, y, ids, GRM_df = load_data(
        base_cfg["paths"],
        target_column=base_cfg.get("target_column", "y_adjusted"),
        standardize_features=base_cfg.get("standardize_features", False),
    )
    # Also load evaluation target (e.g., mean phenotype) for correlation metrics
    eval_target_col = base_cfg.get("eval_target_column", "y_mean")
    _X_eval, y_eval, ids_eval, _ = load_data(
        base_cfg["paths"],
        target_column=eval_target_col,
        standardize_features=base_cfg.get("standardize_features", False),
    )
    n = X.shape[0]
    # Ensure IDs align between training and evaluation targets if provided
    if ids is not None and ids_eval is not None:
        ids_np = np.asarray(ids)
        ids_eval_np = np.asarray(ids_eval)
        if ids_np.shape == ids_eval_np.shape and not np.array_equal(ids_np, ids_eval_np):
            raise ValueError(
                "ID mismatch between training and evaluation targets. Ensure the same ordering of individuals."
            )
    ids = np.asarray(ids) if ids is not None else (np.asarray(ids_eval) if ids_eval is not None else np.arange(n))

    # Outer CV splitter
    kf_outer = KFold(n_splits=outer_folds, shuffle=True, random_state=seed)

    # Accumulators
    oof_pred = np.zeros(n, dtype=np.float32)
    oof_fold = -np.ones(n, dtype=int)
    fold_metrics: List[Dict[str, float]] = []
    best_params_list: List[Dict[str, Any]] = []

    # Outer folds
    for fold_idx, (train_idx, test_idx) in enumerate(kf_outer.split(np.arange(n)), start=1):
        logging.info(
            f"Outer fold {fold_idx}/{outer_folds}: training on {len(train_idx)} samples, testing on {len(test_idx)} samples"
        )

        # Fold data
        X_train_full = X[train_idx]
        y_train_full = y[train_idx]
        y_eval_full = y_eval[train_idx]
        GRM_train = GRM_df.iloc[train_idx, train_idx] if GRM_df is not None else None

        # ---------------------------
        # Inner tuning objective
        # ---------------------------
        def objective(trial: optuna.trial.Trial) -> float:
            cfg_trial = json.loads(json.dumps(base_cfg))

            # Model space
            mspace = search_space.get("model", {})
            hd_choices = mspace.get("hidden_dims_choices", [[128, 64], [256, 128], [64, 64]])
            hd_choices_str = encode_choices_for_optuna(hd_choices)
            hd_choice = trial.suggest_categorical("hidden_dims", hd_choices_str)
            cfg_trial["model"]["hidden_dims"] = decode_choice(hd_choice)
            cfg_trial["model"]["dropout"] = trial.suggest_float("dropout", *mspace.get("dropout_range", (0.0, 0.6)))
            cfg_trial["model"]["batch_norm"] = trial.suggest_categorical("batch_norm", [True, False])

            # Graph space
            gspace = search_space.get("graph", {})
            cfg_trial.setdefault("graph", {})
            cfg_trial["graph"]["source"] = trial.suggest_categorical(
                "source", gspace.get("source_choices", ["snp", "grm"]))
            cfg_trial["graph"]["knn_k"] = trial.suggest_int("knn_k", *gspace.get("knn_k_range", (3, 10)))
            cfg_trial["graph"]["weighted_edges"] = trial.suggest_categorical("weighted_edges", [False, True])
            cfg_trial["graph"]["symmetrize_mode"] = trial.suggest_categorical(
                "symmetrize_mode", gspace.get("symmetrize_mode_choices", ["mutual", "union"])
            )
            cfg_trial["graph"]["laplacian_smoothing"] = trial.suggest_categorical(
                "laplacian_smoothing", gspace.get("laplacian_smoothing_choices", [True, False])
            )
            cfg_trial["graph"]["ensemble_models"] = trial.suggest_int(
                "ensemble_models", *gspace.get("ensemble_models_range", (1, 8))
            )
            cfg_trial["graph"]["val_fraction"] = base_cfg.get("graph", {}).get("val_fraction", 0.1)
            cfg_trial["graph"]["test_fraction"] = base_cfg.get("graph", {}).get("test_fraction", 0.1)

            # Training space
            tspace = search_space.get("training", {})
            cfg_trial.setdefault("training", {})
            cfg_trial["training"]["lr"] = trial.suggest_float(
                "lr", *tspace.get("lr_loguniform", (1e-4, 5e-3)), log=True
            )
            cfg_trial["training"]["weight_decay"] = trial.suggest_float(
                "weight_decay", *tspace.get("wd_loguniform", (1e-7, 1e-3)), log=True
            )
            cfg_trial["training"]["optimizer"] = trial.suggest_categorical(
                "optimizer", tspace.get("optimizer_choices", ["adam", "sgd"])
            )
            cfg_trial["training"]["epochs"] = trial.suggest_int(
                "epochs", *tspace.get("epochs_range", (50, 300)), step=10
            )
            cfg_trial["training"]["patience"] = tspace.get("patience", 0)
            cfg_trial["training"]["loss"] = trial.suggest_categorical(
                "loss", tspace.get("loss_choices", ["mse"])
            )

            # Feature selection space
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

            set_seed(base_cfg.get("seed", 42))

            # Build global adjacency on OUTER-TRAIN ONLY for this trial
            A_global_train = build_global_adjacency(X_train_full, GRM_train, cfg_trial["graph"])

            # Inner splits
            if use_inner_kfold:
                inner_kf = KFold(n_splits=int(cfg.get("inner_folds", 3)), shuffle=True, random_state=seed + fold_idx)
                inner_splits = list(inner_kf.split(np.arange(len(train_idx))))
            else:
                train_loc_idx, val_loc_idx, _ = _select_train_val_indices(
                    len(train_idx), float(inner_val_frac), seed + fold_idx
                )
                inner_splits = [(train_loc_idx, val_loc_idx)]

            fold_r_values: List[float] = []
            fold_best_epochs: List[int] = []

            for inner_train_loc, inner_val_loc in inner_splits:
                # Induced subgraphs (both from OUTER-TRAIN graph)
                A_tr = A_global_train[inner_train_loc][:, inner_train_loc].tocsr()
                A_val = A_global_train[inner_val_loc][:, inner_val_loc].tocsr()

                # Feature selection on inner-train
                if cfg_trial["feature_selection"].get("use_snp_selection", False):
                    k_snps = int(
                        min(cfg_trial["feature_selection"]["num_snps"] or X_train_full.shape[1], X_train_full.shape[1])
                    )
                    cols_sel = _select_top_snps_by_abs_corr(
                        X_train_full[inner_train_loc], y_train_full[inner_train_loc], k_snps
                    )
                else:
                    cols_sel = slice(None)

                # Device
                device = torch.device("cpu")
                if torch.cuda.is_available():
                    ngpu = torch.cuda.device_count()
                    device_id = trial.number % max(1, ngpu)
                    device = torch.device(f"cuda:{device_id}")

                # To torch
                A_tr_idx, A_tr_val, A_tr_shape = to_sparse(A_tr, device)
                A_val_idx, A_val_val, A_val_shape = to_sparse(A_val, device)

                # Loss
                loss_name = cfg_trial["training"].get("loss", "mse").lower()
                loss_fn = nn.L1Loss() if loss_name == "mae" else nn.MSELoss()

                ensemble_count = int(cfg_trial["graph"].get("ensemble_models", 1))
                best_r_this_split = -1.0
                best_epoch_this_split = 1

                if ensemble_count <= 1:
                    # Single model
                    model = GCN(
                        in_dim=X_train_full[inner_train_loc][:, cols_sel].shape[1],
                        hidden_dims=cfg_trial["model"]["hidden_dims"],
                        dropout=cfg_trial["model"]["dropout"],
                        use_bn=cfg_trial["model"]["batch_norm"],
                    ).to(device)
                    optimizer = _optimizer(
                        cfg_trial["training"]["optimizer"],
                        model.parameters(),
                        cfg_trial["training"]["lr"],
                        cfg_trial["training"]["weight_decay"],
                    )
                    Xtr_t = torch.from_numpy(X_train_full[inner_train_loc][:, cols_sel]).to(device)
                    ytr_t = torch.from_numpy(y_train_full[inner_train_loc]).to(device)
                    Xval_t = torch.from_numpy(X_train_full[inner_val_loc][:, cols_sel]).to(device)

                    no_imp = 0
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
                        r_val = _pearson_corr(y_eval_full[inner_val_loc], pred_val)

                        if r_val > best_r_this_split + 1e-12:
                            best_r_this_split = r_val
                            best_epoch_this_split = ep
                            no_imp = 0
                        else:
                            no_imp += 1
                            if cfg_trial["training"]["patience"] > 0 and no_imp >= cfg_trial["training"]["patience"]:
                                break

                else:
                    # Ensemble on inner-train partitions
                    A_train_csr = A_tr
                    parts = partition_train_graph(A_train_csr, ensemble_count)
                    epochs = int(cfg_trial["training"]["epochs"])

                    # Prepare validation features once
                    Xval_t = torch.from_numpy(X_train_full[inner_val_loc][:, cols_sel]).to(device)

                    # We’ll track validation r per epoch by averaging submodels’ preds
                    # (train submodels in lockstep epochs)
                    # Initialize submodels/optimizers
                    submodels: List[GCN] = []
                    opts = []
                    for nodes in parts:
                        if not nodes:
                            submodels.append(None)
                            opts.append(None)
                            continue
                        X_sub = X_train_full[inner_train_loc][nodes][:, cols_sel]
                        model_i = GCN(
                            in_dim=X_sub.shape[1],
                            hidden_dims=cfg_trial["model"]["hidden_dims"],
                            dropout=cfg_trial["model"]["dropout"],
                            use_bn=cfg_trial["model"]["batch_norm"],
                        ).to(device)
                        opt_i = _optimizer(
                            cfg_trial["training"]["optimizer"],
                            model_i.parameters(),
                            cfg_trial["training"]["lr"],
                            cfg_trial["training"]["weight_decay"],
                        )
                        submodels.append(model_i)
                        opts.append(opt_i)

                    # Precompute sparse per partition
                    part_sparse = []
                    part_feats = []
                    part_targets = []
                    for nodes in parts:
                        if not nodes:
                            part_sparse.append(None)
                            part_feats.append(None)
                            part_targets.append(None)
                            continue
                        A_sub = A_train_csr[nodes][:, nodes].tocsr()
                        A_sub_idx, A_sub_val, A_sub_shape = to_sparse(A_sub, device)
                        X_sub = X_train_full[inner_train_loc][nodes][:, cols_sel]
                        y_sub = y_train_full[inner_train_loc][nodes]
                        part_sparse.append((A_sub_idx, A_sub_val, A_sub_shape))
                        part_feats.append(torch.from_numpy(X_sub).to(device))
                        part_targets.append(torch.from_numpy(y_sub).to(device))

                    for ep in range(1, epochs + 1):
                        # Train all partitions one epoch
                        for sm, opt_i, sp, xf, yt in zip(submodels, opts, part_sparse, part_feats, part_targets):
                            if sm is None:
                                continue
                            A_sub_idx, A_sub_val, A_sub_shape = sp
                            sm.train()
                            opt_i.zero_grad()
                            pred_sub = sm(xf, A_sub_idx, A_sub_val, A_sub_shape)
                            loss = loss_fn(pred_sub, yt)
                            loss.backward()
                            opt_i.step()

                        # Validate ensemble
                        with torch.no_grad():
                            preds = []
                            for sm in submodels:
                                if sm is None:
                                    continue
                                sm.eval()
                                preds.append(sm(Xval_t, A_val_idx, A_val_val, A_val_shape))
                            if len(preds) == 0:
                                r_val = -1.0
                            else:
                                preds_stack = torch.stack(preds, dim=1).mean(dim=1).cpu().numpy()
                                r_val = _pearson_corr(y_eval_full[inner_val_loc], preds_stack)

                        if r_val > best_r_this_split + 1e-12:
                            best_r_this_split = r_val
                            best_epoch_this_split = ep

                fold_r_values.append(best_r_this_split)
                fold_best_epochs.append(best_epoch_this_split)

            # Mean inner r and mean best epoch across inner splits
            mean_r = float(np.mean(fold_r_values))
            mean_best_epoch = int(round(float(np.mean(fold_best_epochs)))) if len(fold_best_epochs) > 0 else int(
                cfg_trial["training"]["epochs"]
            )
            trial.set_user_attr("best_epoch", mean_best_epoch)
            return mean_r

        # Study per fold
        sampler = optuna.samplers.TPESampler(
            seed=base_cfg.get("seed", 42),
            n_startup_trials=cfg.get("n_startup_trials", 10),
        )
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=cfg.get("pruner_startup_trials", 5),
            n_warmup_steps=cfg.get("pruner_warmup_epochs", 5),
        )
        study_name = cfg.get("study_name", f"nestedcv_fold{fold_idx}")
        storage = cfg.get("storage", None)
        if storage:
            study = optuna.create_study(
                direction="maximize",
                study_name=study_name,
                storage=storage,
                load_if_exists=True,
                sampler=sampler,
                pruner=pruner,
            )
        else:
            study = optuna.create_study(
                direction="maximize",
                study_name=study_name,
                sampler=sampler,
                pruner=pruner,
            )

        logging.info(f"Starting hyperparameter tuning for fold {fold_idx} (n_trials={cfg.get('n_trials', 40)})...")
        study.optimize(
            objective,
            n_trials=cfg.get("n_trials", 40),
            timeout=cfg.get("timeout_seconds", None),
            n_jobs=int(cfg.get("n_jobs", 1)),
            gc_after_trial=True,
            show_progress_bar=bool(cfg.get("show_progress_bar", False)),
        )

        best_params = study.best_params
        best_epoch = int(study.best_trial.user_attrs.get("best_epoch", base_cfg.get("training", {}).get("epochs", 100)))
        logging.info(f"Fold {fold_idx}: Best inner validation Pearson r = {study.best_value:.4f}")
        logging.info(f"Fold {fold_idx}: Best hyperparameters = {best_params}")
        logging.info(f"Fold {fold_idx}: Using best-epoch = {best_epoch} for final training")
        best_params_list.append({"fold": fold_idx, "best_params": best_params, "best_epoch": best_epoch})

        # Merge best params for final training
        cfg_fold = _merge_best_params(base_cfg, best_params)

        # ---------------------------
        # FINAL TRAINING for this fold
        # Build graphs CONSISTENT with inner scope (no all-nodes graph)
        # ---------------------------
        # A_train: built only from OUTER-TRAIN nodes with best graph params
        A_train = build_global_adjacency(X_train_full, GRM_train, cfg_fold["graph"])
        # A_test: built only from OUTER-TEST nodes with best graph params
        GRM_test = GRM_df.iloc[test_idx, test_idx] if GRM_df is not None else None
        X_test_full = X[test_idx]
        A_test = build_global_adjacency(X_test_full, GRM_test, cfg_fold["graph"])

        # Feature selection on OUTER-TRAIN
        if cfg_fold.get("feature_selection", {}).get("use_snp_selection", False):
            k = int(min(cfg_fold["feature_selection"].get("num_snps", X.shape[1]), X.shape[1]))
            cols_sel = _select_top_snps_by_abs_corr(X_train_full, y_train_full, k)
        else:
            cols_sel = slice(None)

        # To torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        A_tr_idx, A_tr_val, A_tr_shape = to_sparse(A_train, device)
        A_te_idx, A_te_val, A_te_shape = to_sparse(A_test, device)

        # Loss (honor tuned loss)
        loss_name_final = cfg_fold.get("training", {}).get("loss", "mse").lower()
        loss_fn_final = nn.L1Loss() if loss_name_final == "mae" else nn.MSELoss()

        ensemble_count = int(cfg_fold["graph"].get("ensemble_models", 1))
        yhat_test: np.ndarray

        if ensemble_count <= 1:
            # Single model final training with best-epoch
            model = GCN(
                in_dim=X_train_full[:, cols_sel].shape[1],
                hidden_dims=cfg_fold["model"]["hidden_dims"],
                dropout=cfg_fold["model"]["dropout"],
                use_bn=cfg_fold["model"]["batch_norm"],
            ).to(device)
            optimizer = _optimizer(
                cfg_fold["training"]["optimizer"], model.parameters(),
                cfg_fold["training"]["lr"], cfg_fold["training"].get("weight_decay", 0.0)
            )
            X_tr_t = torch.from_numpy(X_train_full[:, cols_sel]).to(device)
            y_tr_t = torch.from_numpy(y_train_full).to(device)

            for ep in range(1, best_epoch + 1):
                model.train()
                optimizer.zero_grad()
                pred = model(X_tr_t, A_tr_idx, A_tr_val, A_tr_shape)
                loss = loss_fn_final(pred, y_tr_t)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                X_te_t = torch.from_numpy(X_test_full[:, cols_sel]).to(device)
                yhat_test = model(X_te_t, A_te_idx, A_te_val, A_te_shape).cpu().numpy()

        else:
            # Ensemble final training with best-epoch
            A_train_csr = A_train
            parts = partition_train_graph(A_train_csr, ensemble_count)
            submodels: List[GCN] = []

            # Prepare per-partition tensors once
            part_sparse = []
            part_feats = []
            part_targets = []
            for nodes in parts:
                if not nodes:
                    part_sparse.append(None)
                    part_feats.append(None)
                    part_targets.append(None)
                    continue
                A_sub = A_train_csr[nodes][:, nodes].tocsr()
                part_sparse.append(to_sparse(A_sub, device))
                part_feats.append(torch.from_numpy(X_train_full[nodes][:, cols_sel]).to(device))
                part_targets.append(torch.from_numpy(y_train_full[nodes]).to(device))

            # Init submodels/opts
            opts = []
            for nodes in parts:
                if not nodes:
                    submodels.append(None)
                    opts.append(None)
                    continue
                model_i = GCN(
                    in_dim=X_train_full[nodes][:, cols_sel].shape[1],
                    hidden_dims=cfg_fold["model"]["hidden_dims"],
                    dropout=cfg_fold["model"]["dropout"],
                    use_bn=cfg_fold["model"]["batch_norm"],
                ).to(device)
                opt_i = _optimizer(
                    cfg_fold["training"]["optimizer"], model_i.parameters(),
                    cfg_fold["training"]["lr"], cfg_fold["training"].get("weight_decay", 0.0)
                )
                submodels.append(model_i)
                opts.append(opt_i)

            # Train for best_epoch
            for ep in range(1, best_epoch + 1):
                for sm, opt_i, sp, xf, yt in zip(submodels, opts, part_sparse, part_feats, part_targets):
                    if sm is None:
                        continue
                    A_sub_idx, A_sub_val, A_sub_shape = sp
                    sm.train()
                    opt_i.zero_grad()
                    pred_sub = sm(xf, A_sub_idx, A_sub_val, A_sub_shape)
                    loss = loss_fn_final(pred_sub, yt)
                    loss.backward()
                    opt_i.step()

            # Ensemble predict on test
            with torch.no_grad():
                X_te_t = torch.from_numpy(X_test_full[:, cols_sel]).to(device)
                preds = []
                for sm in submodels:
                    if sm is None:
                        continue
                    sm.eval()
                    preds.append(sm(X_te_t, A_te_idx, A_te_val, A_te_shape))
                yhat_test = torch.stack(preds, dim=1).mean(dim=1).cpu().numpy() if len(preds) > 0 else \
                    np.zeros(len(test_idx), dtype=np.float32)

        # Store predictions and metric
        oof_pred[test_idx] = yhat_test.flatten()
        oof_fold[test_idx] = fold_idx
        r_test = _pearson_corr(y_eval[test_idx], yhat_test)
        fold_metrics.append({"fold": fold_idx, "pearson_r": float(r_test)})
        logging.info(f"Fold {fold_idx} | Test Pearson r = {r_test:.4f}")

        del study
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Overall OOF Pearson r
    overall_r = _pearson_corr(y_eval, oof_pred)
    logging.info(f"Overall Pearson r (nested CV, {outer_folds}-fold) = {overall_r:.4f}")

    # Save outputs
    out_dir = base_cfg["paths"].get("output_dir", ".")
    os.makedirs(out_dir, exist_ok=True)
    import pandas as pd
    pd.DataFrame(
        {"id": ids, "fold": oof_fold, "y_true": y_eval, "y_pred": oof_pred}
    ).to_csv(os.path.join(out_dir, "nested_oof_predictions.csv"), index=False)

    results = {
        "overall": {"pearson_r": overall_r},
        "per_fold": fold_metrics,
        "best_params_per_fold": best_params_list,
    }
    save_json(results, os.path.join(out_dir, "nested_cv_metrics.json"))
    logging.info(
        f"Saved nested CV predictions to nested_oof_predictions.csv and metrics to nested_cv_metrics.json in {out_dir}"
    )


def _select_train_val_indices(n: int, val_fraction: float, seed: int):
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = int(n * val_fraction)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Nested cross-validation for GCN/GCN-RS with inner hyperparam tuning"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to nested CV config JSON")
    args = parser.parse_args()
    main(args.config)