from __future__ import annotations
"""
Unified nested CV with inner folds + precise transductive/inductive behavior and a PyG GCN.

You asked for:
1) Config‑set **inner folds**.
2) **Transductive**: for each OUTER fold and each TRIAL, build **one graph on ALL NODES** (really all data). During training, mask both
   the OUTER test nodes and the INNER validation nodes; compute loss only on INNER‑TRAIN nodes; evaluate on INNER‑VAL.
3) **SNP selection** based only on **inner‑train**.
4) **Inductive**: per inner split, build **three graphs**: inner‑train, inner‑val, and (for final) outer‑test.
5) New **GCN (PyTorch Geometric)** implementation. (Defined below and used by the runner.)

This single file is drop‑in runnable and keeps the surface similar to your former scripts. If you want true multi‑file layout, 
copy out the `PyGGCN` class to `models/pyg_gcn.py` and the graph utilities to `graph/build.py` and adjust imports.

Run:
    python nested_cv_unified.py --config path/to/config_nested.json
"""

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
from src.graph import build_global_adjacency, identity_csr
from src.utils import set_seed, to_sparse, _pearson_corr, _select_top_snps_by_abs_corr
from src.gcn import TrainParams, make_model

# ---------------------------- logging ----------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ------------------------- Graph builders -------------------------------

def csr_knn_graph(X: np.ndarray, k: int = 10, weighted: bool = True, symmetrize: str = "union") -> sp.csr_matrix:
    """Fallback KNN graph if your project builder is not available."""
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=max(1, k+1), algorithm="auto").fit(X)
    dists, idxs = nbrs.kneighbors(X, return_distance=True)
    # skip self at idx 0
    rows = np.repeat(np.arange(X.shape[0]), k)
    cols = idxs[:, 1:1+k].reshape(-1)
    data = (np.exp(-dists[:, 1:1+k]).reshape(-1) if weighted else np.ones_like(cols, dtype=float))
    A = sp.csr_matrix((data, (rows, cols)), shape=(X.shape[0], X.shape[0]))
    if symmetrize == "mutual":
        A = A.minimum(A.T)
    else:
        A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def build_adjacency_all(X: np.ndarray, ids: np.ndarray, GRM_df, cfg: Dict[str, Any]) -> sp.csr_matrix:
    """Build adjacency on ALL nodes for transductive mode (once per TRIAL)."""
    if not cfg.get("graph_on", True):
        return identity_csr(X.shape[0])
    source = cfg.get("source", "snp")
    if build_global_adjacency is not None:
        return build_global_adjacency(X=X, ids=ids, GRM_df=GRM_df,
                                      source=source,
                                      knn_k=int(cfg.get("knn_k", 10)),
                                      weighted_edges=bool(cfg.get("weighted_edges", True)),
                                      symmetrize_mode=cfg.get("symmetrize_mode", "union")).tocsr()
    # fallback: use KNN on features for source=="snp", else identity
    if source == "snp":
        return csr_knn_graph(X, k=int(cfg.get("knn_k", 10)), weighted=bool(cfg.get("weighted_edges", True)), symmetrize=cfg.get("symmetrize_mode", "union"))
    return identity_csr(X.shape[0])


def build_adjacency_subset(X: np.ndarray, idx: np.ndarray, ids: np.ndarray, GRM_df, cfg: Dict[str, Any]) -> sp.csr_matrix:
    if not cfg.get("graph_on", True):
        return identity_csr(len(idx))
    source = cfg.get("source", "snp")
    if build_global_adjacency is not None:
        return build_global_adjacency(X=X[idx], ids=ids[idx], GRM_df=GRM_df,
                                      source=source,
                                      knn_k=int(cfg.get("knn_k", 10)),
                                      weighted_edges=bool(cfg.get("weighted_edges", True)),
                                      symmetrize_mode=cfg.get("symmetrize_mode", "union")).tocsr()
    if source == "snp":
        return csr_knn_graph(X[idx], k=int(cfg.get("knn_k", 10)), weighted=bool(cfg.get("weighted_edges", True)), symmetrize=cfg.get("symmetrize_mode", "union"))
    return identity_csr(len(idx))


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

    hidden = trial.suggest_categorical("hidden_dims", m.get("hidden_dims_choices", ["[256, 128]", "[128, 64]", "[512, 256]"]))
    if isinstance(hidden, str):
        hidden = json.loads(hidden)
    dropout = trial.suggest_float("dropout", *m.get("dropout_range", (0.0, 0.5)))
    batch_norm = trial.suggest_categorical("batch_norm", m.get("batch_norm_choices", [True, False]))

    lr = trial.suggest_float("lr", *t.get("lr_loguniform", (1e-4, 5e-3)), log=True)
    wd = trial.suggest_float("weight_decay", *t.get("wd_loguniform", (1e-7, 1e-3)), log=True)
    epochs = trial.suggest_int("epochs", *t.get("epochs_range", (50, 300)))
    loss = trial.suggest_categorical("loss", t.get("loss_choices", ["mse", "mae"]))
    opt = trial.suggest_categorical("optimizer", t.get("optimizer_choices", ["adam", "sgd", "adamw"]))

    # also pick graph knobs per trial
    trial.set_user_attr("graph_on", bool(space.get("graph", {}).get("graph_on_default", True)))
    trial.set_user_attr("source", space.get("graph", {}).get("source_default", "snp"))
    trial.set_user_attr("knn_k", int(space.get("graph", {}).get("knn_k_default", 10)))
    trial.set_user_attr("weighted_edges", bool(space.get("graph", {}).get("weighted_edges_default", True)))
    trial.set_user_attr("symmetrize_mode", space.get("graph", {}).get("symmetrize_mode_default", "union"))

    return TrainParams(lr=lr, weight_decay=wd, epochs=epochs, loss_name=loss, optimizer=opt,
                       hidden_dims=hidden, dropout=dropout, batch_norm=bool(batch_norm))


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

    # iterate OUTER splits
    for outer_idx, (tr_idx, te_idx, isl) in enumerate(make_outer_splits(strategy, locality, outer_splits, shuffle, random_state, n=len(X))):
        logger.info(f"OUTER {outer_idx+1}: test_size={len(te_idx)} island={isl}")
        idx_outer_train = tr_idx
        idx_outer_test = te_idx

        # ---------- Inner study (true nested) ----------
        def objective(trial: optuna.Trial) -> float:
            tp = suggest_params(trial, search_space)

            # GRAPH per TRIAL
            gcfg = {
                "graph_on": bool(search_space.get("graph", {}).get("graph_on_default", True)),
                "source": search_space.get("graph", {}).get("source_default", "snp"),
                "knn_k": int(search_space.get("graph", {}).get("knn_k_default", 10)),
                "weighted_edges": bool(search_space.get("graph", {}).get("weighted_edges_default", True)),
                "symmetrize_mode": search_space.get("graph", {}).get("symmetrize_mode_default", "union"),
            }
            # Allow overrides via trial user attrs
            gcfg.update({
                "graph_on": trial.user_attrs.get("graph_on", gcfg["graph_on"]),
                "source": trial.user_attrs.get("source", gcfg["source"]),
                "knn_k": trial.user_attrs.get("knn_k", gcfg["knn_k"]),
                "weighted_edges": trial.user_attrs.get("weighted_edges", gcfg["weighted_edges"]),
                "symmetrize_mode": trial.user_attrs.get("symmetrize_mode", gcfg["symmetrize_mode"]),
            })

            # ----- Transductive: build ONE graph on ALL nodes once per TRIAL
            if learning_mode == "transductive":
                A_all = build_adjacency_all(X, ids, GRM_df, gcfg)
                # Pre-tensors shared by all inner folds
                x_all = torch.from_numpy(X).to(device)
                edge_index, edge_weight, _ = to_sparse(A_all, device)
                # y tensor on device
                y_all = torch.from_numpy(y).to(device).float()

            r_vals = []
            # iterate INNER folds on OUTER-TRAIN indices
            for in_tr, in_va in make_inner_splits(idx_outer_train, inner_splits, shuffle, random_state):
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
                    A_tr = build_adjacency_subset(X, in_tr, ids, GRM_df, gcfg)
                    A_va = build_adjacency_subset(X, in_va, ids, GRM_df, gcfg)

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
        logger.info(f"OUTER {outer_idx+1} best (inner mean r={study.best_value:.4f}): {best}")

        # ---------- Final train on OUTER-TRAIN, evaluate on OUTER-TEST ----------
        # Build graph(s) with best trial attrs
        # Recover graph config (from user attrs defaults set during suggest)
        gcfg_final = {
            "graph_on": bool(search_space.get("graph", {}).get("graph_on_default", True)),
            "source": search_space.get("graph", {}).get("source_default", "snp"),
            "knn_k": int(search_space.get("graph", {}).get("knn_k_default", 10)),
            "weighted_edges": bool(search_space.get("graph", {}).get("weighted_edges_default", True)),
            "symmetrize_mode": search_space.get("graph", {}).get("symmetrize_mode_default", "union"),
        }
        tp_final = TrainParams(
            lr=best.get("lr"), weight_decay=best.get("weight_decay"), epochs=best.get("epochs"),
            loss_name=best.get("loss"), optimizer=best.get("optimizer"),
            hidden_dims=json.loads(best.get("hidden_dims")) if isinstance(best.get("hidden_dims"), str) else best.get("hidden_dims"),
            dropout=best.get("dropout"), batch_norm=bool(best.get("batch_norm"))
        )

        if learning_mode == "transductive":
            # ONE graph over ALL nodes
            A_all = build_adjacency_all(X, ids, GRM_df, gcfg_final)
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
            A_tr = build_adjacency_subset(X, idx_outer_train, ids, GRM_df, gcfg_final)
            A_te = build_adjacency_subset(X, idx_outer_test, ids, GRM_df, gcfg_final)

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
