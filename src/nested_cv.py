from __future__ import annotations
import argparse
import json
import logging
import os
import gc
from typing import Any, Dict, Optional
from sklearn.model_selection import KFold

import numpy as np
import optuna
import torch
import torch.nn as nn
import scipy.sparse as sp


# ------------------- project helpers (kept optional) -------------------
from src.data import load_data
from src.graph import build_adjacency
from src.utils import set_seed, to_sparse, _pearson_corr, _select_top_snps_by_abs_corr, encode_choices_for_optuna, decode_choice
from src.gcn import TrainParams, make_model

# ---------------------------- logging ----------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ------------------------- Island naming --------------------------------
# Map known island numeric codes to human-readable names.
ISLAND_ID_TO_NAME: Dict[int, str] = {
    20: "Nesøy",
    22: "Myken",
    23: "Træna",
    24: "Selvær",
    26: "Gjerøy",
    27: "Hestmannøy",
    28: "Indre Kvarøy",
    33: "Onøy og Lurøy",
    34: "Lovund",
    35: "Sleneset",
    38: "Aldra",
    # Southern islands grouped/renamed
    60: "Southern 1",
    61: "Southern 2",
    63: "Southern 3",
    67: "Southern 4",
    68: "Southern 5",
}

def _island_label(isl_id: Optional[int], code_to_label: Optional[Dict[int, str]]) -> str:
    if isl_id is None:
        return "None"
    try:
        isl_int = int(isl_id)
    except Exception:
        return str(isl_id)
    if isl_int in ISLAND_ID_TO_NAME:
        return ISLAND_ID_TO_NAME[isl_int]
    if code_to_label and isl_int in code_to_label:
        return ISLAND_ID_TO_NAME[int(code_to_label[isl_int])]
    return str(isl_int)


# ---------------------------- CV helpers --------------------------------

def make_outer_splits(strategy: str, locality: np.ndarray, n_splits: int, shuffle: bool, random_state: int, n: int):
    if strategy == "leave_island_out":
        uniq = np.unique(locality)
        idx_all = np.arange(n)
        for isl in uniq:
            te_idx = np.where(locality == isl)[0]
            tr_idx = np.setdiff1d(idx_all, te_idx, assume_unique=False)
            yield (tr_idx, te_idx, int(isl))
    else:
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        for tr, te in kf.split(np.arange(n)):
            yield (tr, te, None)


def make_inner_splits(idx_train: np.ndarray, n_splits: int, shuffle: bool, random_state: int):
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    for tr, va in kf.split(idx_train):
        yield (idx_train[tr], idx_train[va])


def make_inner_loio_splits(locality: np.ndarray, idx_outer_train: np.ndarray):
    """Inner LOIO within the outer-train set: one inner fold per island present in outer-train.

    Returns a list of (train_idx, val_idx, island_id) tuples, where:
    - val_idx contains all samples from a single island within the outer-train set
    - train_idx contains all other samples (the remaining islands in outer-train)
    """
    loc_tr = locality[idx_outer_train]
    uniq = np.unique(loc_tr)
    splits = []
    for isl in uniq:
        val_mask = (loc_tr == isl)
        val_idx = idx_outer_train[val_mask]
        train_idx = idx_outer_train[~val_mask]
        splits.append((train_idx, val_idx, int(isl)))
    return splits



def make_optimizer(name: str, params, lr: float, wd: float):
    name = (name or "adam").lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    return torch.optim.Adam(params, lr=lr, weight_decay=wd)


def make_loss(name: str):
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


# ------------------------- Objectives (inner loop) ----------------------

def suggest_params(trial: optuna.Trial, space: Dict[str, Any]) -> TrainParams:
    m = space.get("model", {})
    t = space.get("training", {})
    g = space.get("graph", {})

    hidden = m.get("hidden_dims_choices", [])
    hidden = encode_choices_for_optuna(hidden)
    hidden = trial.suggest_categorical("hidden_dims", hidden)

    dropout = trial.suggest_float("dropout", *m.get("dropout_range", (0.0, 0.5)))
    batch_norm = trial.suggest_categorical("batch_norm", m.get("batch_norm_choices", [True, False]))

    lr = trial.suggest_float("lr", *t.get("lr_loguniform", (1e-4, 5e-3)), log=True)
    wd = trial.suggest_float("weight_decay", *t.get("wd_loguniform", (1e-7, 1e-3)), log=True)
    epochs = trial.suggest_int("epochs", *t.get("epochs_range", (50, 300)))
    loss = trial.suggest_categorical("loss", t.get("loss_choices", ["mse", "mae"]))
    opt = trial.suggest_categorical("optimizer", t.get("optimizer_choices", ["adam", "sgd", "adamw"]))

    # Graph hyperparameters suggested by trial so Optuna logs them
    graph_on = trial.suggest_categorical("graph_on", g.get("graph_on_choices", [True, False]))
    if graph_on:
        source = trial.suggest_categorical("source", g.get("source_choices", ["snp", "grm"]))
        knn_k = trial.suggest_int("knn_k", *g.get("knn_k_range", (5, 30)))
        weighted_edges = trial.suggest_categorical("weighted_edges", g.get("weighted_edges_choices", [True, False]))
        symmetrize_mode = trial.suggest_categorical("symmetrize_mode", g.get("symmetrize_mode_choices", ["union", "mutual"]))
        # no-op: suggestions exist in trial.params; model construction uses only training params
        _ = (source, knn_k, weighted_edges, symmetrize_mode)

    return TrainParams(lr=lr, weight_decay=wd, epochs=epochs, loss_name=loss, optimizer=opt,
                       hidden_dims=decode_choice(hidden), dropout=dropout, batch_norm=bool(batch_norm))


# --------------------------- Runner (nested) ----------------------------

def run_nested_cv(config: Dict[str, Any]):
    base = config["base_train"]
    search_space = config.get("search_space", {})

    seed = int(base.get("seed", 42))
    set_seed(seed)

    # ---- Load data
    if load_data is None:
        raise RuntimeError("load_data() not found. Please provide your project loader via src.data.load_data.")

    X, y, ids, GRM_df, locality, code_to_label, y_eval = load_data(
        base["paths"],
        target_column=base.get("target_column", "y_adjusted"),
        standardize_features=base.get("standardize_features", False),
        return_locality=True,
        min_count=20,
        return_eval=True,
        eval_target_column=base.get("eval_target_column", "y_mean"),
    )
    if y_eval is None:
        y_eval = y.copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ---- CV config
    cv_cfg = config.get("cv", {})
    strategy = cv_cfg.get("strategy", "kfold").lower()  # "kfold" or "leave_island_out"
    outer_splits = int(cv_cfg.get("n_splits", 10))
    inner_splits = int(cv_cfg.get("inner_splits", 5))
    shuffle = bool(cv_cfg.get("shuffle", True))
    random_state = int(cv_cfg.get("random_state", seed))

    learning_mode = config.get("learning_mode", "transductive").lower()  # "transductive" or "inductive"

    # ---- Optuna global knobs
    n_trials = int(config.get("n_trials", 100))
    enable_pruning = bool(config.get("enable_pruning", True))
    pruner = (
        optuna.pruners.MedianPruner(n_warmup_steps=int(config.get("pruner_warmup_epochs", 5)))
        if enable_pruning else optuna.pruners.NopPruner()
    )

    outer_results = []
    best_params_per_fold = []

    # iterate OUTER splits
    for outer_idx, (tr_idx, te_idx, isl) in enumerate(make_outer_splits(strategy, locality, outer_splits, shuffle, random_state, n=len(X))):
        isl_name = _island_label(isl, code_to_label)
        logger.info(f"OUTER {outer_idx+1}: test_size={len(te_idx)} island={isl} ({isl_name})")
        idx_outer_train = tr_idx
        idx_outer_test = te_idx

        if strategy == "leave_island_out":
            inner_isls = np.unique(locality[idx_outer_train])
            inner_names = [_island_label(int(i), code_to_label) for i in inner_isls]
            pairs = ", ".join(f"{int(i)}({n})" for i, n in zip(inner_isls, inner_names))
            logger.info(f"OUTER {outer_idx+1}: inner LOIO validation islands: {pairs}")

        # ---------- Inner study (true nested) ----------
        def objective(trial: optuna.Trial) -> float:
            tp = suggest_params(trial, search_space)

            # GRAPH per TRIAL from suggested params
            gspace = search_space.get("graph", {})
            gcfg = {
                "graph_on": bool(trial.params.get("graph_on", gspace.get("graph_on_default", True))),
                "source": trial.params.get("source", gspace.get("source_default", "snp")),
                "knn_k": int(trial.params.get("knn_k", gspace.get("knn_k_default", 10))),
                "weighted_edges": bool(trial.params.get("weighted_edges", gspace.get("weighted_edges_default", True))),
                "symmetrize_mode": trial.params.get("symmetrize_mode", gspace.get("symmetrize_mode_default", "union")),
            }

            # ----- Transductive: build ONE graph on ALL nodes once per TRIAL
            if learning_mode == "transductive":
                A_all = build_adjacency(X, GRM_df, gcfg, node_idx=None)
                # Pre-tensors shared by all inner folds
                x_all = torch.from_numpy(X).to(device)
                edge_index, edge_weight, _ = to_sparse(A_all, device)
                # y tensor on device
                y_all = torch.from_numpy(y).to(device).float()

            r_vals = []
            # iterate INNER folds on OUTER-TRAIN indices
            if strategy == "leave_island_out":
                inner_plan = make_inner_loio_splits(locality, idx_outer_train)
            else:
                inner_plan = [(tr, va, None) for (tr, va) in make_inner_splits(idx_outer_train, inner_splits, shuffle, random_state)]

            for in_tr, in_va, in_isl in inner_plan:
                if learning_mode == "transductive":
                    # Masking: loss computed only on inner-train; outer-test and inner-val are masked implicitly
                    model = make_model(in_dim=X.shape[1], tp=tp).to(device)
                    opt = make_optimizer(tp.optimizer, model.parameters(), lr=tp.lr, wd=tp.weight_decay)
                    loss_fn = make_loss(tp.loss_name)

                    train_masked_epochs(model, x_all, edge_index, edge_weight, y_all,
                                        train_idx=in_tr, epochs=tp.epochs, opt=opt, loss_fn=loss_fn)
                    model.eval()
                    with torch.no_grad():
                        yhat = model(x_all, edge_index, edge_weight).detach().cpu().numpy().ravel()
                    r_vals.append(_pearson_corr(y_eval[in_va], yhat[in_va]))

                else:  # INDUCTIVE
                    # FS on INNER-TRAIN only
                    cols = slice(None)
                    if search_space.get("feature_selection", {}).get("use_snp_selection_default", False):
                        k = int(search_space.get("feature_selection", {}).get("num_snps_default", 20000))
                        cols = _select_top_snps_by_abs_corr(X[in_tr], y[in_tr], min(k, X.shape[1]))

                    X_tr, X_va = X[in_tr][:, cols], X[in_va][:, cols]
                    # Three graphs: inner-train, inner-val, outer-test (test only needed later; we stick to spec and build val graph now)
                    A_tr = build_adjacency(X, GRM_df, gcfg, node_idx=in_tr)
                    A_va = build_adjacency(X, GRM_df, gcfg, node_idx=in_va)

                    # PyG edge_index per subset
                    def csr_to_edge_index(A: sp.csr_matrix, base: int = 0):
                        coo = A.tocoo()
                        ei = torch.tensor(np.vstack([coo.row, coo.col]) + base, dtype=torch.long, device=device)
                        ew = torch.tensor(coo.data, dtype=torch.float32, device=device)
                        return ei, ew

                    ei_tr, ew_tr = csr_to_edge_index(A_tr)
                    ei_va, ew_va = csr_to_edge_index(A_va)
                    x_tr = torch.from_numpy(X_tr).to(device)
                    y_tr_t = torch.from_numpy(y[in_tr]).to(device).float()
                    x_va = torch.from_numpy(X_va).to(device)

                    model = make_model(in_dim=X_tr.shape[1], tp=tp).to(device)
                    opt = make_optimizer(tp.optimizer, model.parameters(), lr=tp.lr, wd=tp.weight_decay)
                    loss_fn = make_loss(tp.loss_name)

                    # standard inductive train (train graph only)
                    for _ in range(int(tp.epochs)):
                        model.train()
                        opt.zero_grad()
                        pred = model(x_tr, ei_tr, ew_tr)
                        loss = loss_fn(pred, y_tr_t)
                        loss.backward()
                        opt.step()

                    model.eval()
                    with torch.no_grad():
                        yhat_va = model(x_va, ei_va, ew_va).detach().cpu().numpy().ravel()
                    r_vals.append(_pearson_corr(y_eval[in_va], yhat_va))

                # cleanup per inner fold
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            return float(np.mean(r_vals)) if r_vals else 0.0

        study = optuna.create_study(direction="maximize",
                                    study_name=f"inner_outer{outer_idx}",
                                    sampler=optuna.samplers.TPESampler(seed=seed),
                                    pruner=pruner)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=bool(config.get("show_progress_bar", True)))
        best = study.best_params
        # Decode complex params (e.g., hidden_dims)
        best_decoded = dict(best)
        if "hidden_dims" in best_decoded:
            try:
                best_decoded["hidden_dims"] = decode_choice(best_decoded["hidden_dims"])  # type: ignore[arg-type]
            except Exception:
                pass
        full_best = best_decoded
        logger.info(f"OUTER {outer_idx+1} best (inner mean r={study.best_value:.4f}): {full_best}")
        best_params_per_fold.append({
            "fold": int(outer_idx + 1),
            "best_params": full_best,
            "mean_inner_r": float(study.best_value),
        })

        # ---------- Final train on OUTER-TRAIN, evaluate on OUTER-TEST ----------
        # Build graph(s) with best trial params (fallback to defaults)
        gspace = search_space.get("graph", {})
        gcfg_final = {
            "graph_on": bool(best.get("graph_on", gspace.get("graph_on_default", True))),
            "source": best.get("source", gspace.get("source_default", "snp")),
            "knn_k": int(best.get("knn_k", gspace.get("knn_k_default", 10))),
            "weighted_edges": bool(best.get("weighted_edges", gspace.get("weighted_edges_default", True))),
            "symmetrize_mode": best.get("symmetrize_mode", gspace.get("symmetrize_mode_default", "union")),
        }
        tp_final = TrainParams(
            lr=best.get("lr"), weight_decay=best.get("weight_decay"), epochs=best.get("epochs"),
            loss_name=best.get("loss"), optimizer=best.get("optimizer"),
            hidden_dims=json.loads(best.get("hidden_dims")) if isinstance(best.get("hidden_dims"), str) else best.get("hidden_dims"),
            dropout=best.get("dropout"), batch_norm=bool(best.get("batch_norm"))
        )

        if learning_mode == "transductive":
            # ONE graph over ALL nodes
            A_all = build_adjacency(X, GRM_df, gcfg_final, node_idx=None)
            edge_index, edge_weight, _ = to_sparse(A_all, device)
            x_all = torch.from_numpy(X).to(device)
            y_all = torch.from_numpy(y).to(device).float()

            model = make_model(in_dim=X.shape[1], tp=tp_final).to(device)
            opt = make_optimizer(tp_final.optimizer, model.parameters(), lr=tp_final.lr, wd=tp_final.weight_decay)
            loss_fn = make_loss(tp_final.loss_name)

            # Final train uses all OUTER-TRAIN nodes (no inner masking now), OUTER-TEST stays masked.
            train_masked_epochs(model, x_all, edge_index, edge_weight, y_all,
                                train_idx=idx_outer_train, epochs=tp_final.epochs, opt=opt, loss_fn=loss_fn)
            model.eval()
            with torch.no_grad():
                yhat_all = model(x_all, edge_index, edge_weight).detach().cpu().numpy().ravel()
            r_test = _pearson_corr(y_eval[idx_outer_test], yhat_all[idx_outer_test])

        else:  # INDUCTIVE final: train on OUTER-TRAIN graph, eval on OUTER-TEST graph
            # FS refit on OUTER-TRAIN only if enabled
            cols = slice(None)
            if search_space.get("feature_selection", {}).get("use_snp_selection_default", False):
                k = int(search_space.get("feature_selection", {}).get("num_snps_default", 20000))
                cols = _select_top_snps_by_abs_corr(X[idx_outer_train], y[idx_outer_train], min(k, X.shape[1]))

            X_tr, X_te = X[idx_outer_train][:, cols], X[idx_outer_test][:, cols]
            A_tr = build_adjacency(X, GRM_df, gcfg_final, node_idx=idx_outer_train)
            A_te = build_adjacency(X, GRM_df, gcfg_final, node_idx=idx_outer_test)

            def csr_to_edge_index(A: sp.csr_matrix):
                coo = A.tocoo()
                ei = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long, device=device)
                ew = torch.tensor(coo.data, dtype=torch.float32, device=device)
                return ei, ew

            ei_tr, ew_tr = csr_to_edge_index(A_tr)
            ei_te, ew_te = csr_to_edge_index(A_te)
            x_tr = torch.from_numpy(X_tr).to(device)
            y_tr_t = torch.from_numpy(y[idx_outer_train]).to(device).float()
            x_te = torch.from_numpy(X_te).to(device)

            model = make_model(in_dim=X_tr.shape[1], tp=tp_final).to(device)
            opt = make_optimizer(tp_final.optimizer, model.parameters(), lr=tp_final.lr, wd=tp_final.weight_decay)
            loss_fn = make_loss(tp_final.loss_name)

            for _ in range(int(tp_final.epochs)):
                model.train()
                opt.zero_grad()
                pred = model(x_tr, ei_tr, ew_tr)
                loss = loss_fn(pred, y_tr_t)
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                yhat_te = model(x_te, ei_te, ew_te).detach().cpu().numpy().ravel()
            r_test = _pearson_corr(y_eval[idx_outer_test], yhat_te)

        logger.info(f"OUTER {outer_idx+1} TEST r = {r_test:.4f}")
        outer_results.append(float(r_test))

        # cleanup outer
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # ---- save summary
    out_dir = base["paths"].get("output_dir", "outputs/nested_cv")
    out_name = base["paths"].get("output_name", "nested_cv_unified")
    os.makedirs(out_dir, exist_ok=True)
    summary = {
        "mode": learning_mode,
        "cv_strategy": strategy,
        "outer_test_corr": outer_results,
        "outer_test_corr_mean": float(np.mean(outer_results)) if outer_results else None,
        "outer_test_corr_std": float(np.std(outer_results)) if outer_results else None,
        "inner_splits": inner_splits,
        "outer_splits": outer_splits,
        "best_params_per_fold": best_params_per_fold,
    }
    with open(os.path.join(out_dir, f"{out_name}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"DONE. Mean OUTER r = {summary['outer_test_corr_mean']:.4f} ± {summary['outer_test_corr_std']:.4f}")


# ------------------------------ CLI ------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Unified nested CV (inner folds) with transductive/inductive control + PyG GCN")
    ap.add_argument("--config", required=True, type=str)
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    run_nested_cv(cfg)


if __name__ == "__main__":
    main()
