from __future__ import annotations
import argparse
import json
import os
import gc
import logging
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
import torch
import torch.nn as nn
import scipy.sparse as sp

from src.data import load_data
from src.graph import build_global_adjacency  # used when graph_on=True
from src.gcn import GCN
from src.utils import (
    set_seed, to_sparse, save_json, _optimizer,
    _select_top_snps_by_abs_corr, _pearson_corr,
    encode_choices_for_optuna, decode_choice,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ----------------------------- helpers --------------------------------
def _merge_best_params(cfg: Dict[str, Any], best: Dict[str, Any]) -> Dict[str, Any]:
    out = json.loads(json.dumps(cfg))
    g = out.setdefault("graph", {})
    for key in ["graph_on","source","knn_k","weighted_edges","symmetrize_mode","laplacian_smoothing"]:
        if key in best:
            g[key] = best[key]
    m = out.setdefault("model", {})
    if "hidden_dims" in best:
        m["hidden_dims"] = json.loads(best["hidden_dims"]) if isinstance(best["hidden_dims"], str) else best["hidden_dims"]
    for key in ["dropout","batch_norm"]:
        if key in best:
            m[key] = best[key]
    t = out.setdefault("training", {})
    for key in ["lr","weight_decay","optimizer","epochs","patience","loss"]:
        if key in best:
            t[key] = best[key]
    fs = out.setdefault("feature_selection", {})
    if "use_snp_selection" in best:
        fs["use_snp_selection"] = bool(best["use_snp_selection"])
    if fs.get("use_snp_selection", False) and "num_snps" in best:
        fs["num_snps"] = int(best["num_snps"])
    return out

def _identity_adj(n: int) -> sp.csr_matrix:
    """Sparse identity adjacency (self-loops only)."""
    return sp.eye(n, format="csr")

def _make_adjacency(X_sub: np.ndarray,
                    GRM_sub,  # pd.DataFrame or None
                    graph_cfg: Dict[str, Any]) -> sp.csr_matrix:
    """Build adjacency for a subset, honoring graph_on switch."""
    if not graph_cfg.get("graph_on", True):
        return _identity_adj(X_sub.shape[0])
    # When graph_on=True, delegate to your normal builder (uses cfg: source, knn_k, etc.)
    return build_global_adjacency(X_sub, GRM_sub, graph_cfg)

def _outer_loio_splits(locality: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """Return list of (train_idx, test_idx, test_island)."""
    uniq = np.unique(locality)
    splits = []
    all_idx = np.arange(len(locality))
    for isl in uniq:
        test_idx = np.where(locality == isl)[0]
        train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=False)
        splits.append((train_idx, test_idx, int(isl)))
    return splits

def _inner_loio_splits(locality_train: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """LOIOCV within the outer-train set."""
    uniq = np.unique(locality_train)
    idx_all = np.arange(len(locality_train))
    splits = []
    for isl in uniq:
        val_idx = np.where(locality_train == isl)[0]
        tr_idx = np.setdiff1d(idx_all, val_idx, assume_unique=False)
        splits.append((tr_idx, val_idx, int(isl)))
    return splits

# ----------------------------- main -----------------------------------
def main(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    base_cfg = cfg["base_train"]
    search_space = cfg.get("search_space", {})
    n_trials = int(cfg.get("n_trials", 40))

    seed = base_cfg.get("seed", 42)
    set_seed(seed)

    # Load data (train target + evaluation target in one call)
    logging.info("Loading data...")
    eval_target_col = base_cfg.get("eval_target_column", "y_mean")
    X, y, ids, GRM_df, locality, code_to_label, y_eval = load_data(
        base_cfg["paths"],
        target_column=base_cfg.get("target_column", "y_adjusted"),
        standardize_features=base_cfg.get("standardize_features", False),
        return_locality=True,
        min_count=20,
        return_eval=True,
        eval_target_column=eval_target_col,
    )
    n = X.shape[0]

    ids = np.asarray(ids) if ids is not None else np.arange(n)

    outer_splits = _outer_loio_splits(locality)
    logging.info(f"LOIOCV: found {len(outer_splits)} islands (outer folds). Islands={sorted({isl for *_, isl in outer_splits})}")

    # Accumulators
    oof_pred = np.zeros(n, dtype=np.float32)
    oof_fold = -np.ones(n, dtype=int)
    per_fold: List[Dict[str, Any]] = []
    best_params_all: List[Dict[str, Any]] = []

    # --------------------------- Outer loop ----------------------------
    for fold_idx, (train_idx, test_idx, test_island) in enumerate(outer_splits, start=1):
        logging.info(
            f"[Outer {fold_idx}/{len(outer_splits)}] Test island={test_island} "
            f"({len(test_idx)} samples). Train={len(train_idx)} samples."
        )

        X_tr_full = X[train_idx]
        y_tr_full = y[train_idx]
        y_eval_tr_full = y_eval[train_idx]
        GRM_tr = GRM_df.iloc[train_idx, train_idx] if GRM_df is not None else None
        loc_tr = locality[train_idx]

        # ------------------------ Objective ----------------------------
        def objective(trial: optuna.trial.Trial) -> float:
            cfg_trial = json.loads(json.dumps(base_cfg))

            # ---- Model space
            mspace = search_space.get("model", {})
            hd_choices = mspace.get("hidden_dims_choices", [[128,64],[256,128],[512,256]])
            hd_choices_str = encode_choices_for_optuna(hd_choices)
            hd_choice = trial.suggest_categorical("hidden_dims", hd_choices_str)
            cfg_trial["model"]["hidden_dims"] = decode_choice(hd_choice)
            cfg_trial["model"]["dropout"] = trial.suggest_float("dropout", *mspace.get("dropout_range",(0.0,0.6)))
            cfg_trial["model"]["batch_norm"] = trial.suggest_categorical("batch_norm", [True, False])

            # ---- Graph space with gate
            gspace = search_space.get("graph", {})
            cfg_trial.setdefault("graph", {})
            graph_on = trial.suggest_categorical("graph_on", gspace.get("graph_on_choices",[True, False]))
            cfg_trial["graph"]["graph_on"] = graph_on
            if graph_on:
                cfg_trial["graph"]["source"] = trial.suggest_categorical("source", gspace.get("source_choices", ["snp","grm"]))
                # Ensure ≥1
                lo, hi = gspace.get("knn_k_range",(3,10))
                lo = max(1, int(lo))
                cfg_trial["graph"]["knn_k"] = trial.suggest_int("knn_k", lo, int(hi))
                cfg_trial["graph"]["weighted_edges"] = trial.suggest_categorical("weighted_edges",[False, True])
                cfg_trial["graph"]["symmetrize_mode"] = trial.suggest_categorical(
                    "symmetrize_mode", gspace.get("symmetrize_mode_choices", ["mutual","union"])
                )
                cfg_trial["graph"]["laplacian_smoothing"] = trial.suggest_categorical(
                    "laplacian_smoothing", gspace.get("laplacian_smoothing_choices", [True, False])
                )
            else:
                # No graph parameters sampled; builder will return identity adjacency
                cfg_trial["graph"].update({
                    "source": "none",
                    "knn_k": 0,
                    "weighted_edges": False,
                    "symmetrize_mode": "union",
                    "laplacian_smoothing": False,
                })

            # ---- Training space
            tspace = search_space.get("training", {})
            cfg_trial.setdefault("training", {})
            cfg_trial["training"]["lr"] = trial.suggest_float("lr", *tspace.get("lr_loguniform",(1e-4,5e-3)), log=True)
            cfg_trial["training"]["weight_decay"] = trial.suggest_float("weight_decay", *tspace.get("wd_loguniform",(1e-7,1e-3)), log=True)
            cfg_trial["training"]["optimizer"] = trial.suggest_categorical("optimizer", tspace.get("optimizer_choices", ["adam","sgd"]))
            cfg_trial["training"]["epochs"] = trial.suggest_int("epochs", *tspace.get("epochs_range",(50,300)), step=1)
            cfg_trial["training"]["patience"] = int(tspace.get("patience", 0))
            cfg_trial["training"]["loss"] = trial.suggest_categorical("loss", tspace.get("loss_choices", ["mse"]))

            # ---- Feature selection space
            fspace = search_space.get("feature_selection", {})
            use_sel = trial.suggest_categorical("use_snp_selection", fspace.get("use_snp_selection_choices",[False, True]))
            cfg_trial.setdefault("feature_selection", {})
            cfg_trial["feature_selection"]["use_snp_selection"] = bool(use_sel)
            if cfg_trial["feature_selection"]["use_snp_selection"]:
                ns_min, ns_max = fspace.get("num_snps_range",(5000,65000))
                step = int(fspace.get("num_snps_step",5000))
                cfg_trial["feature_selection"]["num_snps"] = trial.suggest_int("num_snps", int(ns_min), int(ns_max), step=step)
            else:
                cfg_trial["feature_selection"]["num_snps"] = None

            set_seed(seed + trial.number)

            # Build a single global adjacency on OUTER-TRAIN (respects graph_on)
            t0 = time.perf_counter()
            A_global_train = _make_adjacency(X_tr_full, GRM_tr, cfg_trial["graph"])
            t1 = time.perf_counter()
            logger.info(
                f"[Outer {fold_idx}] Built train graph (graph_on={cfg_trial['graph'].get('graph_on', True)}, "
                f"source={cfg_trial['graph'].get('source')}, knn_k={cfg_trial['graph'].get('knn_k')}) "
                f"in {t1 - t0:.2f}s | shape={A_global_train.shape}, nnz={A_global_train.nnz}"
            )

            # Inner LOIOCV across remaining islands in training set
            inner_splits = _inner_loio_splits(loc_tr)

            r_values, best_epochs = [], []
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"[Outer {fold_idx}] Using device: {device}")

            for tr_loc_idx, val_loc_idx, val_island in inner_splits:
                # Induced subgraphs from the same outer-train graph
                A_tr = A_global_train[tr_loc_idx][:, tr_loc_idx].tocsr()
                A_val = A_global_train[val_loc_idx][:, val_loc_idx].tocsr()

                # Feature selection on inner-train
                if cfg_trial["feature_selection"].get("use_snp_selection", False):
                    k_snps = int(min(cfg_trial["feature_selection"]["num_snps"] or X_tr_full.shape[1], X_tr_full.shape[1]))
                    cols_sel = _select_top_snps_by_abs_corr(X_tr_full[tr_loc_idx], y_tr_full[tr_loc_idx], k_snps)
                    logger.info(f"[Outer {fold_idx}] Inner FS: selected top {k_snps} SNPs for island {val_island}")
                else:
                    cols_sel = slice(None)

                # To torch sparse
                A_tr_idx, A_tr_val, A_tr_shape = to_sparse(A_tr, device)
                A_val_idx, A_val_val, A_val_shape = to_sparse(A_val, device)

                # Model & loss
                loss_name = cfg_trial["training"].get("loss","mse").lower()
                loss_fn = nn.L1Loss() if loss_name == "mae" else nn.MSELoss()

                model = GCN(
                    in_dim=X_tr_full[tr_loc_idx][:, cols_sel].shape[1],
                    hidden_dims=cfg_trial["model"]["hidden_dims"],
                    dropout=cfg_trial["model"]["dropout"],
                    use_bn=cfg_trial["model"]["batch_norm"],
                ).to(device)
                opt = _optimizer(cfg_trial["training"]["optimizer"], model.parameters(),
                                 cfg_trial["training"]["lr"], cfg_trial["training"]["weight_decay"])

                Xtr_t = torch.from_numpy(X_tr_full[tr_loc_idx][:, cols_sel]).to(device)
                ytr_t = torch.from_numpy(y_tr_full[tr_loc_idx]).to(device)
                Xval_t = torch.from_numpy(X_tr_full[val_loc_idx][:, cols_sel]).to(device)

                best_r, best_ep, no_imp = -1.0, 1, 0
                for ep in range(1, int(cfg_trial["training"]["epochs"]) + 1):
                    model.train()
                    opt.zero_grad()
                    pred_tr = model(Xtr_t, A_tr_idx, A_tr_val, A_tr_shape)
                    loss = loss_fn(pred_tr, ytr_t)
                    loss.backward()
                    opt.step()

                    model.eval()
                    with torch.no_grad():
                        pred_val = model(Xval_t, A_val_idx, A_val_val, A_val_shape).cpu().numpy()
                    r_val = _pearson_corr(y_eval_tr_full[val_loc_idx], pred_val)

                    if r_val > best_r + 1e-12:
                        best_r, best_ep, no_imp = r_val, ep, 0
                    else:
                        no_imp += 1
                        if cfg_trial["training"]["patience"] > 0 and no_imp >= cfg_trial["training"]["patience"]:
                            break

                r_values.append(best_r)
                best_epochs.append(best_ep)

            mean_r = float(np.mean(r_values)) if r_values else -1.0
            mean_best_epoch = int(round(float(np.mean(best_epochs)))) if best_epochs else int(cfg_trial["training"]["epochs"])
            trial.set_user_attr("best_epoch", mean_best_epoch)
            return mean_r

        # ---- Optuna per-outer-fold
        sampler = optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=cfg.get("n_startup_trials", 10),
        )
        if cfg.get("enable_pruning", True):
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=cfg.get("pruner_startup_trials", 5),
                n_warmup_steps=cfg.get("pruner_warmup_epochs", 5),
            )
        else:
            pruner = optuna.pruners.NopPruner()
        study = optuna.create_study(
            direction="maximize",
            study_name=cfg.get("study_name", f"loio_fold{fold_idx}"),
            sampler=sampler,
            pruner=pruner,
            storage=cfg.get("storage", None),
            load_if_exists=bool(cfg.get("storage", None)),
        )

        logging.info(f"Starting tuning (outer fold {fold_idx}, island={test_island}) with n_trials={n_trials} ...")
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=cfg.get("timeout_seconds", None),
            n_jobs=int(cfg.get("n_jobs", 1)),
            gc_after_trial=True,
            show_progress_bar=bool(cfg.get("show_progress_bar", False)),
        )

        best_params = study.best_params
        best_epoch = int(study.best_trial.user_attrs.get("best_epoch", base_cfg.get("training", {}).get("epochs", 100)))
        logging.info(f"Outer {fold_idx} (island={test_island}) | Best inner r = {study.best_value:.4f}")
        logging.info(f"Outer {fold_idx} | Best hyperparameters = {best_params}")
        logging.info(f"Outer {fold_idx} | Using best_epoch = {best_epoch}")
        best_params_all.append({"fold": fold_idx, "island": test_island, "best_params": best_params, "best_epoch": best_epoch})

        # ---------------------- Final train on outer-train ----------------------
        cfg_fold = _merge_best_params(base_cfg, best_params)

        # Build A_train on outer-train and A_test on outer-test
        t0 = time.perf_counter()
        A_train = _make_adjacency(X_tr_full, GRM_tr, cfg_fold["graph"])
        t1 = time.perf_counter()
        logger.info(
            f"[Outer {fold_idx}] Built final train graph in {t1 - t0:.2f}s | shape={A_train.shape}, nnz={A_train.nnz}"
        )
        GRM_te = GRM_df.iloc[test_idx, test_idx] if GRM_df is not None else None
        X_te_full = X[test_idx]
        t2 = time.perf_counter()
        A_test = _make_adjacency(X_te_full, GRM_te, cfg_fold["graph"])
        t3 = time.perf_counter()
        logger.info(
            f"[Outer {fold_idx}] Built test graph in {t3 - t2:.2f}s | shape={A_test.shape}, nnz={A_test.nnz}"
        )

        # Feature selection on outer-train
        if cfg_fold.get("feature_selection", {}).get("use_snp_selection", False):
            k = int(min(cfg_fold["feature_selection"].get("num_snps", X.shape[1]), X.shape[1]))
            cols_sel = _select_top_snps_by_abs_corr(X_tr_full, y_tr_full, k)
            logger.info(f"[Outer {fold_idx}] Final FS: selected top {k} SNPs")
        else:
            cols_sel = slice(None)

        # Torch tensors
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        A_tr_idx, A_tr_val, A_tr_shape = to_sparse(A_train, device)
        A_te_idx, A_te_val, A_te_shape = to_sparse(A_test, device)

        loss_name_final = cfg_fold.get("training", {}).get("loss","mse").lower()
        loss_fn_final = nn.L1Loss() if loss_name_final == "mae" else nn.MSELoss()

        model = GCN(
            in_dim=X_tr_full[:, cols_sel].shape[1],
            hidden_dims=cfg_fold["model"]["hidden_dims"],
            dropout=cfg_fold["model"]["dropout"],
            use_bn=cfg_fold["model"]["batch_norm"],
        ).to(device)
        opt = _optimizer(cfg_fold["training"]["optimizer"], model.parameters(),
                         cfg_fold["training"]["lr"], cfg_fold["training"].get("weight_decay", 0.0))

        X_tr_t = torch.from_numpy(X_tr_full[:, cols_sel]).to(device)
        y_tr_t = torch.from_numpy(y_tr_full).to(device)

        logger.info(f"[Outer {fold_idx}] Training final model for {best_epoch} epochs")
        for ep in range(1, best_epoch + 1):
            model.train()
            opt.zero_grad()
            pred = model(X_tr_t, A_tr_idx, A_tr_val, A_tr_shape)
            loss = loss_fn_final(pred, y_tr_t)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            X_te_t = torch.from_numpy(X_te_full[:, cols_sel]).to(device)
            yhat_test = model(X_te_t, A_te_idx, A_te_val, A_te_shape).cpu().numpy()

        # Store outer-fold predictions & metric
        oof_pred[test_idx] = yhat_test.flatten()
        oof_fold[test_idx] = fold_idx
        r_test = _pearson_corr(y_eval[test_idx], yhat_test)
        per_fold.append({"fold": fold_idx, "test_island": int(test_island), "pearson_r": float(r_test)})
        logging.info(f"[Outer {fold_idx}] island={test_island} | Test Pearson r = {r_test:.4f}")

        del study, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    overall_r = _pearson_corr(y_eval, oof_pred)
    logging.info(f"Overall OOF Pearson r (LOIO nested CV) = {overall_r:.4f}")

    # Save
    out_dir = base_cfg["paths"].get("output_dir", ".")
    out_suffix = cfg.get("phenotype", "[unknown]")
    os.makedirs(out_dir, exist_ok=True)
    import pandas as pd
    pd.DataFrame({"id": ids, "fold": oof_fold, "island": locality, "y_true": y_eval, "y_pred": oof_pred}) \
      .to_csv(os.path.join(out_dir, f"loio_nested_oof_predictions{out_suffix}.csv"), index=False)

    results = {
        "overall": {"pearson_r": overall_r},
        "per_fold": per_fold,
        "best_params_per_fold": best_params_all,
        "islands": sorted(list(map(int, np.unique(locality)))),
        "island_code_to_label": code_to_label
    }
    save_json(results, os.path.join(out_dir, f"loio_nested_cv_metrics{out_suffix}.json"))
    logging.info("Saved LOIOCV outputs.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Leave-one-island-out nested CV (outer LOIO, inner LOIO)")
    p.add_argument("--config", type=str, required=True, help="Path to nested CV config JSON")
    args = p.parse_args()
    main(args.config)
