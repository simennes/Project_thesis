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
from src.utils import set_seed, to_sparse, _pearson_corr, _select_top_snps_by_abs_corr, encode_choices_for_optuna, decode_choice, _optimizer
from src.models import TrainParams, make_model

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

def make_outer_splits(strategy: str, locality: np.ndarray, n_splits: int, shuffle: bool, random_state: int, n: int, 
                      predefined_folds: Optional[list] = None, ids: Optional[np.ndarray] = None):
    if strategy == "leave_island_out":
        # One outer fold per unique island code
        uniq = np.unique(locality)
        for isl in uniq:
            te = np.where(locality == isl)[0]
            tr = np.where(locality != isl)[0]
            yield (tr, te, int(isl))
    elif strategy == "kfold" and predefined_folds is not None:
        # Use predefined folds from JSON file
        if ids is None:
            raise ValueError("IDs must be provided when using predefined folds")
        
        # Convert ids to strings for matching (JSON IDs are strings)
        id_to_idx = {str(id_val): idx for idx, id_val in enumerate(ids)}
        
        for fold_idx, test_ids in enumerate(predefined_folds):
            # Map test IDs to indices
            te = []
            for test_id in test_ids:
                if str(test_id) in id_to_idx:
                    te.append(id_to_idx[str(test_id)])
            
            te = np.array(te, dtype=int)
            tr = np.array([i for i in range(n) if i not in te], dtype=int)
            
            if len(te) == 0:
                logger.warning(f"Fold {fold_idx + 1}: No test samples found. Skipping this fold.")
                continue
                
            yield (tr, te, None)
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



# use shared _optimizer from utils


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
    fsel = space.get("feature_selection", {})

    model_type = m.get("model_type_default", "gcn")
    model_type = trial.suggest_categorical("model_type", m.get("model_type_choices", ["gcn", "gat", "graphsage"]))

    hidden = m.get("hidden_dims_choices", [])
    hidden = encode_choices_for_optuna(hidden)
    hidden = trial.suggest_categorical("hidden_dims", hidden)

    dropout = trial.suggest_float("dropout", *m.get("dropout_range", (0.0, 0.5)))
    batch_norm = trial.suggest_categorical("batch_norm", m.get("batch_norm_choices", [True, False]))

    # GAT-specific params (conditionally sampled)
    gat_heads = None
    gat_attn_dropout = None
    gat_concat_hidden = None
    if model_type == "gat":
        gat_heads = trial.suggest_int("gat_heads", *m.get("gat_heads_range", (1, 8)))
        gat_attn_dropout = trial.suggest_float("gat_attn_dropout", *m.get("gat_attn_dropout_range", (0.0, 0.6)))
        gat_concat_hidden = trial.suggest_categorical("gat_concat_hidden", m.get("gat_concat_hidden_choices", [True, False]))

    lr = trial.suggest_float("lr", *t.get("lr_loguniform", (1e-4, 5e-3)), log=True)
    wd = trial.suggest_float("weight_decay", *t.get("wd_loguniform", (1e-7, 1e-3)), log=True)
    epochs = trial.suggest_int("epochs", *t.get("epochs_range", (50, 300)))
    loss = trial.suggest_categorical("loss", t.get("loss_choices", ["mse", "mae"]))
    opt = trial.suggest_categorical("optimizer", t.get("optimizer_choices", ["adam", "sgd", "adamw"]))

    # Graph hyperparameters suggested by trial so Optuna logs them
    graph_mode = trial.suggest_categorical("graph_mode", g.get("graph_mode_choices", ["knn", "grm", "cutoff", "off"]))
    if graph_mode == "knn":
        source = trial.suggest_categorical("source", g.get("source_choices", ["snp", "grm"]))
        knn_k = trial.suggest_int("knn_k", *g.get("knn_k_range", (5, 30)))
        weighted_edges = trial.suggest_categorical("weighted_edges", g.get("weighted_edges_choices", [True, False]))
        symmetrize_mode = trial.suggest_categorical("symmetrize_mode", g.get("symmetrize_mode_choices", ["union", "mutual"]))
        _ = (source, knn_k, weighted_edges, symmetrize_mode)
    elif graph_mode == "grm":
        grm_norm = trial.suggest_categorical("grm_norm", g.get("grm_norm_choices", ["gcn", "cosine", "none", "cosine_then_gcn"]))
        self_loops = trial.suggest_categorical("self_loops", g.get("self_loops_choices", [True, False]))
        _ = (grm_norm, self_loops)
    elif graph_mode == "cutoff":
        cutoff = trial.suggest_float("cutoff", *g.get("cutoff_range", (0.2, 0.9)))
        grm_norm = trial.suggest_categorical("grm_norm", g.get("grm_norm_choices", ["none", "cosine", "gcn", "cosine_then_gcn"]))
        self_loops = trial.suggest_categorical("self_loops", g.get("self_loops_choices", [True, False]))
        _ = (cutoff, grm_norm, self_loops)

    # Feature selection knobs (logged in trial; used outside TrainParams)
    use_fs = trial.suggest_categorical(
        "use_snp_selection", fsel.get("use_snp_selection_choices", [False, True])
    )
    if use_fs:
        # number of SNPs to select if feature selection is on
        _ = trial.suggest_int("num_snps", *fsel.get("num_snps_range", (2000, 60000)))

    return TrainParams(lr=lr, weight_decay=wd, epochs=epochs, loss_name=loss, optimizer=opt,
                       hidden_dims=decode_choice(hidden), dropout=dropout, batch_norm=bool(batch_norm),
                       model_type=model_type, gat_heads=gat_heads, gat_attn_dropout=gat_attn_dropout,
                       gat_concat_hidden=gat_concat_hidden)


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

    # ---- Optional island inclusion filter (by original location number)
    cv_cfg = config.get("cv", {})
    include_islands = cv_cfg.get("include_islands")
    if include_islands:
        # Normalize input to a flat python list
        if isinstance(include_islands, (list, tuple, set, np.ndarray)):
            include_list = list(include_islands)
        else:
            include_list = [include_islands]
        # Convert numpy scalars to native python types
        include_list = [x.item() if isinstance(x, np.generic) else x for x in include_list]

        # Build label->code map from code_to_label (which is code->label)
        label_to_code = {str(v): int(k) for k, v in (code_to_label or {}).items()}
        present_codes = set(np.unique(locality).astype(int).tolist())

        # Resolve requested include list into encoded codes
        include_codes = set()
        for val in include_list:
            # Try matching by original label string
            sval = str(val)
            if sval in label_to_code:
                include_codes.add(int(label_to_code[sval]))
                continue
            # Else, if it's already an encoded code number
            try:
                ival = int(val)
                if ival in present_codes:
                    include_codes.add(ival)
            except Exception:
                pass

        if not include_codes:
            available = [f"{c}:{(code_to_label or {}).get(int(c), '?')}" for c in sorted(present_codes)]
            raise ValueError(
                f"include_islands={include_islands} did not match any samples after mapping. "
                f"Available codes/labels: {available}"
            )

        mask = np.isin(locality.astype(int), np.fromiter(include_codes, dtype=int))
        idx = np.where(mask)[0]
        if idx.size == 0:
            available = [f"{c}:{(code_to_label or {}).get(int(c), '?')}" for c in sorted(present_codes)]
            raise ValueError(
                f"include_islands={include_islands} filtered out all samples. "
                f"Matched codes={sorted(include_codes)}. Available codes/labels: {available}"
            )

        # Apply filtering consistently across all aligned arrays
        X = X[idx]
        y = y[idx]
        y_eval = y_eval[idx]
        ids = ids[idx] if ids is not None else None
        locality = locality[idx]
        if GRM_df is not None:
            GRM_df = GRM_df.iloc[idx, idx]

        # Log human-readable info
        kept_codes = sorted(set(locality.astype(int).tolist()))
        kept_labels = [(code_to_label or {}).get(int(c), str(c)) for c in kept_codes]
        logger.info(
            "Filtered to %d samples from islands (codes->labels): %s based on include_islands=%s",
            idx.size,
            ", ".join(f"{c}->{lbl}" for c, lbl in zip(kept_codes, kept_labels)),
            include_islands,
        )

    # ---- CV config
    cv_cfg = config.get("cv", {})
    strategy = cv_cfg.get("strategy", "kfold").lower()  # "kfold" or "leave_island_out"
    outer_splits = int(cv_cfg.get("n_splits", 10))
    inner_splits = int(cv_cfg.get("inner_splits", 5))
    shuffle = bool(cv_cfg.get("shuffle", True))
    random_state = int(cv_cfg.get("random_state", seed))
    
    # Load predefined folds if path is provided
    predefined_folds = None
    predefined_folds_path = cv_cfg.get("predefined_folds_path", None)
    if predefined_folds_path and strategy == "kfold":
        logger.info(f"Loading predefined folds from: {predefined_folds_path}")
        with open(predefined_folds_path, "r", encoding="utf-8") as f:
            predefined_folds = json.load(f)
        outer_splits = len(predefined_folds)
        logger.info(f"Loaded {outer_splits} predefined folds")

    # Optional: run only selected outer split indices (1-based)
    sel_from_cfg = config.get("selected_splits", None)
    sel_from_cv = cv_cfg.get("selected_splits", None)
    selected_splits = sel_from_cfg if sel_from_cfg is not None else sel_from_cv
    if isinstance(selected_splits, (list, tuple, np.ndarray)):
        try:
            selected_splits = [int(x) for x in selected_splits]
        except Exception:
            selected_splits = None
    elif isinstance(selected_splits, (str,)):
        s = selected_splits.strip().lower()
        if s in ("false", "none", "", "0"):
            selected_splits = None
        else:
            try:
                parsed = json.loads(selected_splits)
                if isinstance(parsed, list):
                    selected_splits = [int(x) for x in parsed]
                else:
                    selected_splits = None
            except Exception:
                # try comma-separated
                try:
                    selected_splits = [int(x) for x in selected_splits.split(",") if x.strip()]
                except Exception:
                    selected_splits = None
    else:
        selected_splits = None

    selected_set = set(selected_splits) if selected_splits else None
    if selected_set:
        logger.info("Running only selected outer splits: %s (1-based)", sorted(selected_set))

    learning_mode = config.get("learning_mode", "transductive").lower()  # "transductive", "inductive", or "inductive_ensemble"

    # ---- Optuna global knobs
    n_trials = int(config.get("n_trials", 100))
    enable_pruning = bool(config.get("enable_pruning", True))
    pruner = (
        optuna.pruners.MedianPruner(n_warmup_steps=int(config.get("pruner_warmup_epochs", 5)))
        if enable_pruning else optuna.pruners.NopPruner()
    )

    outer_results = []
    outer_results_weighted = []  # for inductive_ensemble weighted by inter-island similarity
    best_params_per_fold = []

    # iterate OUTER splits
    for outer_idx, (tr_idx, te_idx, isl) in enumerate(make_outer_splits(strategy, locality, outer_splits, shuffle, random_state, n=len(X), 
                                                                         predefined_folds=predefined_folds, ids=ids)):
        # Filter by selected_splits if provided (1-based indices)
        if selected_set and (outer_idx + 1) not in selected_set:
            continue
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
        if learning_mode == "inductive_ensemble":
            if strategy != "leave_island_out":
                raise ValueError("inductive_ensemble requires cv.strategy='leave_island_out'.")

            # Map each island present in OUTER-TRAIN to its indices
            inner_isls = np.unique(locality[idx_outer_train]).astype(int)
            inner_isls = [int(i) for i in inner_isls]
            isl_to_idx = {
                int(i): idx_outer_train[np.where(locality[idx_outer_train] == int(i))[0]]
                for i in inner_isls
            }

            gspace = search_space.get("graph", {})

            per_island_best = []

            # Optimize a model per INNER island i (train on island i, validate across other islands in OUTER-TRAIN)
            for isl_i in inner_isls:
                def objective_island(trial: optuna.Trial) -> float:
                    tp = suggest_params(trial, search_space)

                    # Graph config from trial
                    graph_mode = trial.params.get("graph_mode", gspace.get("graph_mode_default", "knn"))
                    gcfg = {"graph_mode": graph_mode}
                    if graph_mode == "knn":
                        gcfg.update({
                            "source": trial.params.get("source", gspace.get("source_default", "snp")),
                            "knn_k": int(trial.params.get("knn_k", gspace.get("knn_k_default", 10))),
                            "weighted_edges": bool(trial.params.get("weighted_edges", gspace.get("weighted_edges_default", True))),
                            "symmetrize_mode": trial.params.get("symmetrize_mode", gspace.get("symmetrize_mode_default", "union")),
                        })
                    elif graph_mode == "grm":
                        gcfg.update({
                            "grm_norm": trial.params.get("grm_norm", gspace.get("grm_norm_default", "gcn")),
                            "self_loops": bool(trial.params.get("self_loops", gspace.get("self_loops_default", True))),
                        })
                    elif graph_mode == "cutoff":
                        gcfg.update({
                            "cutoff": float(trial.params.get("cutoff", gspace.get("cutoff_default", 0.5))),
                            "grm_norm": trial.params.get("grm_norm", gspace.get("grm_norm_default", "none")),
                            "self_loops": bool(trial.params.get("self_loops", gspace.get("self_loops_default", True))),
                        })

                    # Feature selection based on island i (train split) — LOIO ensemble rule
                    cols = slice(None)
                    if bool(trial.params.get("use_snp_selection", False)):
                        k = int(trial.params.get("num_snps", X.shape[1]))
                        cols = _select_top_snps_by_abs_corr(
                            X[isl_to_idx[isl_i]], y[isl_to_idx[isl_i]], min(k, X.shape[1])
                        )

                    # Build train tensors for island i
                    A_i = build_adjacency(X, GRM_df, gcfg, node_idx=isl_to_idx[isl_i])
                    coo_i = A_i.tocoo()
                    ei_i = torch.tensor(np.vstack([coo_i.row, coo_i.col]), dtype=torch.long, device=device)
                    ew_i = torch.tensor(coo_i.data, dtype=torch.float32, device=device)
                    x_i = torch.from_numpy(X[isl_to_idx[isl_i]][:, cols]).to(device)
                    y_i_t = torch.from_numpy(y[isl_to_idx[isl_i]]).to(device).float()

                    model = make_model(in_dim=x_i.shape[1], tp=tp).to(device)
                    opt = _optimizer(tp.optimizer, model.parameters(), tp.lr, tp.weight_decay)
                    loss_fn = make_loss(tp.loss_name)

                    # Train on island i only
                    for _ in range(int(tp.epochs)):
                        model.train()
                        opt.zero_grad()
                        pred = model(x_i, ei_i, ew_i)
                        loss = loss_fn(pred, y_i_t)
                        loss.backward()
                        opt.step()

                    # Evaluate on each other island j in OUTER-TRAIN
                    r_js = []
                    for isl_j in inner_isls:
                        if int(isl_j) == int(isl_i):
                            continue
                        idx_j = isl_to_idx[int(isl_j)]
                        if idx_j.size == 0:
                            continue
                        A_j = build_adjacency(X, GRM_df, gcfg, node_idx=idx_j)
                        coo_j = A_j.tocoo()
                        ei_j = torch.tensor(np.vstack([coo_j.row, coo_j.col]), dtype=torch.long, device=device)
                        ew_j = torch.tensor(coo_j.data, dtype=torch.float32, device=device)
                        x_j = torch.from_numpy(X[idx_j][:, cols]).to(device)
                        with torch.no_grad():
                            yhat_j = model(x_j, ei_j, ew_j).detach().cpu().numpy().ravel()
                        r_js.append(_pearson_corr(y_eval[idx_j], yhat_j))

                    # cleanup
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                    return float(np.mean(r_js)) if r_js else 0.0

                study_i = optuna.create_study(direction="maximize",
                                              study_name=f"inner_outer{outer_idx}_isl{isl_i}",
                                              sampler=optuna.samplers.TPESampler(seed=seed),
                                              pruner=pruner)
                study_i.optimize(objective_island, n_trials=n_trials, show_progress_bar=bool(config.get("show_progress_bar", True)))

                best_i = dict(study_i.best_params)
                if "hidden_dims" in best_i:
                    try:
                        best_i["hidden_dims"] = decode_choice(best_i["hidden_dims"])  # type: ignore[arg-type]
                    except Exception:
                        pass
                per_island_best.append({
                    "island_id": int(isl_i),
                    "island_label": _island_label(int(isl_i), code_to_label),
                    "best_params": best_i,
                    "mean_inner_r": float(study_i.best_value),
                })
                logger.info(f"OUTER {outer_idx+1} | island {isl_i} best inner mean r={study_i.best_value:.4f} params={best_i}")

            # ---------- Final ensemble: train per-island model on its island; predict on OUTER-TEST; average predictions ----------
            preds_test_collect = []
            for best_entry in per_island_best:
                isl_i = int(best_entry["island_id"])
                best = best_entry["best_params"]

                # Rehydrate TrainParams and graph config
                gcfg_final = {"graph_mode": best.get("graph_mode", gspace.get("graph_mode_default", "knn"))}
                if gcfg_final["graph_mode"] == "knn":
                    gcfg_final.update({
                        "source": best.get("source", gspace.get("source_default", "snp")),
                        "knn_k": int(best.get("knn_k", gspace.get("knn_k_default", 10))),
                        "weighted_edges": bool(best.get("weighted_edges", gspace.get("weighted_edges_default", True))),
                        "symmetrize_mode": best.get("symmetrize_mode", gspace.get("symmetrize_mode_default", "union")),
                    })
                elif gcfg_final["graph_mode"] == "grm":
                    gcfg_final.update({
                        "grm_norm": best.get("grm_norm", gspace.get("grm_norm_default", "gcn")),
                        "self_loops": bool(best.get("self_loops", gspace.get("self_loops_default", True))),
                    })
                elif gcfg_final["graph_mode"] == "cutoff":
                    gcfg_final.update({
                        "cutoff": float(best.get("cutoff", gspace.get("cutoff_default", 0.5))),
                        "grm_norm": best.get("grm_norm", gspace.get("grm_norm_default", "none")),
                        "self_loops": bool(best.get("self_loops", gspace.get("self_loops_default", True))),
                    })

                tp_final = TrainParams(
                    lr=best.get("lr"), weight_decay=best.get("weight_decay"), epochs=best.get("epochs"),
                    loss_name=best.get("loss"), optimizer=best.get("optimizer"),
                    hidden_dims=best.get("hidden_dims"),
                    dropout=best.get("dropout"), batch_norm=bool(best.get("batch_norm")),
                    model_type=best.get("model_type", search_space.get("model", {}).get("model_type_default", "gcn")),
                    gat_heads=best.get("gat_heads", None),
                    gat_attn_dropout=best.get("gat_attn_dropout", None),
                    gat_concat_hidden=best.get("gat_concat_hidden", None),
                )

                # FS based on island i (use best params from island study)
                cols = slice(None)
                if bool(best.get("use_snp_selection", False)):
                    k = int(best.get("num_snps", X.shape[1]))
                    cols = _select_top_snps_by_abs_corr(
                        X[isl_to_idx[isl_i]], y[isl_to_idx[isl_i]], min(k, X.shape[1])
                    )

                # Train on island i
                A_tr = build_adjacency(X, GRM_df, gcfg_final, node_idx=isl_to_idx[isl_i])
                coo_tr = A_tr.tocoo()
                ei_tr = torch.tensor(np.vstack([coo_tr.row, coo_tr.col]), dtype=torch.long, device=device)
                ew_tr = torch.tensor(coo_tr.data, dtype=torch.float32, device=device)
                x_tr = torch.from_numpy(X[isl_to_idx[isl_i]][:, cols]).to(device)
                y_tr_t = torch.from_numpy(y[isl_to_idx[isl_i]]).to(device).float()

                model = make_model(in_dim=x_tr.shape[1], tp=tp_final).to(device)
                opt = _optimizer(tp_final.optimizer, model.parameters(), tp_final.lr, tp_final.weight_decay)
                loss_fn = make_loss(tp_final.loss_name)
                for _ in range(int(tp_final.epochs)):
                    model.train()
                    opt.zero_grad()
                    pred = model(x_tr, ei_tr, ew_tr)
                    loss = loss_fn(pred, y_tr_t)
                    loss.backward()
                    opt.step()

                # Predict on OUTER-TEST island graph
                A_te = build_adjacency(X, GRM_df, gcfg_final, node_idx=idx_outer_test)
                coo_te = A_te.tocoo()
                ei_te = torch.tensor(np.vstack([coo_te.row, coo_te.col]), dtype=torch.long, device=device)
                ew_te = torch.tensor(coo_te.data, dtype=torch.float32, device=device)
                x_te = torch.from_numpy(X[idx_outer_test][:, cols]).to(device)
                model.eval()
                with torch.no_grad():
                    yhat_te = model(x_te, ei_te, ew_te).detach().cpu().numpy().ravel()
                preds_test_collect.append(yhat_te)

                # cleanup per island model
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            # Ensemble by simple mean (unweighted)
            if not preds_test_collect:
                r_test = 0.0
                r_test_weighted = 0.0
            else:
                Y_stack = np.vstack(preds_test_collect)  # shape: (n_models, n_test)
                yhat_ensemble = np.mean(Y_stack, axis=0)
                r_test = _pearson_corr(y_eval[idx_outer_test], yhat_ensemble)

                # Weighted ensemble using inter-island relatedness with softmax (tau=0.003)
                tau = 0.003
                weights_means = []
                for best_entry in per_island_best:
                    isl_i = int(best_entry["island_id"])
                    idx_j = isl_to_idx[int(isl_i)]
                    if GRM_df is None or idx_outer_test.size == 0 or idx_j.size == 0:
                        weights_means.append(np.nan)
                        continue
                    sub = GRM_df.values[np.ix_(idx_outer_test, idx_j)]
                    m = float(np.nanmean(sub)) if sub.size else np.nan
                    weights_means.append(m)

                w = np.array(weights_means, dtype=float)
                if not np.isfinite(w).any():
                    # fallback to uniform if all invalid
                    w = np.ones(len(weights_means), dtype=float) / max(1, len(weights_means))
                else:
                    # replace non-finite with min finite for stability
                    finite_vals = w[np.isfinite(w)]
                    fill_val = finite_vals.min() if finite_vals.size else 0.0
                    w = np.where(np.isfinite(w), w, fill_val)
                    w_stable = (w - np.max(w)) / max(tau, 1e-8)
                    e = np.exp(w_stable)
                    s = e.sum()
                    w = e / s if s != 0 else np.full_like(w, 1.0 / len(w))

                yhat_weighted = np.sum((w[:, None]) * Y_stack, axis=0)
                r_test_weighted = _pearson_corr(y_eval[idx_outer_test], yhat_weighted)

            logger.info(f"OUTER {outer_idx+1} TEST r (ensemble-mean) = {r_test:.4f}; weighted (softmax tau=0.003) = {r_test_weighted:.4f}")
            outer_results.append(float(r_test))
            outer_results_weighted.append(float(r_test_weighted))

            best_params_per_fold.append({
                "fold": int(outer_idx + 1),
                "ensemble_per_island": per_island_best,
                "r_test_mean": float(r_test),
                "r_test_weighted": float(r_test_weighted),
            })

            # cleanup outer fold temps
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # proceed to next OUTER fold
            continue

        # default path (transductive / inductive)
        def objective(trial: optuna.Trial) -> float:
            tp = suggest_params(trial, search_space)

            # GRAPH per TRIAL from suggested params
            gspace = search_space.get("graph", {})
            # Determine mode with defaults
            graph_mode = trial.params.get("graph_mode", gspace.get("graph_mode_default", "knn"))
            gcfg = {"graph_mode": graph_mode}
            if graph_mode == "knn":
                gcfg.update({
                    "source": trial.params.get("source", gspace.get("source_default", "snp")),
                    "knn_k": int(trial.params.get("knn_k", gspace.get("knn_k_default", 10))),
                    "weighted_edges": bool(trial.params.get("weighted_edges", gspace.get("weighted_edges_default", True))),
                    "symmetrize_mode": trial.params.get("symmetrize_mode", gspace.get("symmetrize_mode_default", "union")),
                })
            elif graph_mode == "grm":
                gcfg.update({
                    "grm_norm": trial.params.get("grm_norm", gspace.get("grm_norm_default", "gcn")),
                    "self_loops": bool(trial.params.get("self_loops", gspace.get("self_loops_default", True))),
                })
            elif graph_mode == "cutoff":
                gcfg.update({
                    "cutoff": float(trial.params.get("cutoff", gspace.get("cutoff_default", 0.5))),
                    "grm_norm": trial.params.get("grm_norm", gspace.get("grm_norm_default", "none")),
                    "self_loops": bool(trial.params.get("self_loops", gspace.get("self_loops_default", True))),
                })

            # ----- Transductive: build ONE graph on ALL nodes once per TRIAL
            if learning_mode == "transductive":
                A_all = build_adjacency(X, GRM_df, gcfg, node_idx=None)
                try:
                    nnz = int(A_all.nnz)
                    n_nodes = int(A_all.shape[0])
                    density = float(nnz) / max(1, n_nodes * n_nodes)
                    logger.info(f"Graph built (mode={graph_mode}): nodes={n_nodes}, edges={nnz}, density={density:.6f}")
                except Exception:
                    pass
                # Edges shared by all inner folds
                edge_index, edge_weight, _ = to_sparse(A_all, device)
                # y tensor on device
                y_all_full = torch.from_numpy(y).to(device).float()

            r_vals = []
            # iterate INNER folds on OUTER-TRAIN indices
            if strategy == "leave_island_out":
                inner_plan = make_inner_loio_splits(locality, idx_outer_train)
            else:
                inner_plan = [(tr, va, None) for (tr, va) in make_inner_splits(idx_outer_train, inner_splits, shuffle, random_state)]

            for in_tr, in_va, in_isl in inner_plan:
                if learning_mode == "transductive":
                    # Feature selection on INNER-TRAIN only (avoids leakage)
                    cols = slice(None)
                    if bool(trial.params.get("use_snp_selection", False)):
                        k = int(trial.params.get("num_snps", X.shape[1]))
                        cols = _select_top_snps_by_abs_corr(X[in_tr], y[in_tr], min(k, X.shape[1]))

                    x_all_fold = torch.from_numpy(X[:, cols]).to(device)
                    # Masking: loss computed only on inner-train; outer-test and inner-val are masked implicitly
                    model = make_model(in_dim=x_all_fold.shape[1], tp=tp).to(device)
                    opt = _optimizer(tp.optimizer, model.parameters(), tp.lr, tp.weight_decay)
                    loss_fn = make_loss(tp.loss_name)

                    train_masked_epochs(model, x_all_fold, edge_index, edge_weight, y_all_full,
                                        train_idx=in_tr, epochs=tp.epochs, opt=opt, loss_fn=loss_fn)
                    model.eval()
                    with torch.no_grad():
                        yhat = model(x_all_fold, edge_index, edge_weight).detach().cpu().numpy().ravel()
                    r_vals.append(_pearson_corr(y_eval[in_va], yhat[in_va]))

                else:  # INDUCTIVE
                    # FS on INNER-TRAIN only
                    cols = slice(None)
                    if bool(trial.params.get("use_snp_selection", False)):
                        k = int(trial.params.get("num_snps", X.shape[1]))
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
                    opt = _optimizer(tp.optimizer, model.parameters(), tp.lr, tp.weight_decay)
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
        graph_mode_best = best.get("graph_mode", gspace.get("graph_mode_default", "knn"))
        gcfg_final = {"graph_mode": graph_mode_best}
        if graph_mode_best == "knn":
            gcfg_final.update({
                "source": best.get("source", gspace.get("source_default", "snp")),
                "knn_k": int(best.get("knn_k", gspace.get("knn_k_default", 10))),
                "weighted_edges": bool(best.get("weighted_edges", gspace.get("weighted_edges_default", True))),
                "symmetrize_mode": best.get("symmetrize_mode", gspace.get("symmetrize_mode_default", "union")),
            })
        elif graph_mode_best == "grm":
            gcfg_final.update({
                "grm_norm": best.get("grm_norm", gspace.get("grm_norm_default", "gcn")),
                "self_loops": bool(best.get("self_loops", gspace.get("self_loops_default", True))),
            })
        elif graph_mode_best == "cutoff":
            gcfg_final.update({
                "cutoff": float(best.get("cutoff", gspace.get("cutoff_default", 0.5))),
                "grm_norm": best.get("grm_norm", gspace.get("grm_norm_default", "none")),
                "self_loops": bool(best.get("self_loops", gspace.get("self_loops_default", True))),
            })
        tp_final = TrainParams(
            lr=best.get("lr"), weight_decay=best.get("weight_decay"), epochs=best.get("epochs"),
            loss_name=best.get("loss"), optimizer=best.get("optimizer"),
            hidden_dims=json.loads(best.get("hidden_dims")) if isinstance(best.get("hidden_dims"), str) else best.get("hidden_dims"),
            dropout=best.get("dropout"), batch_norm=bool(best.get("batch_norm")),
            model_type=best.get("model_type", search_space.get("model", {}).get("model_type_default", "gcn")),
            gat_heads=best.get("gat_heads", None),
            gat_attn_dropout=best.get("gat_attn_dropout", None),
            gat_concat_hidden=best.get("gat_concat_hidden", None),
        )

        if learning_mode == "transductive":
            # ONE graph over ALL nodes
            A_all = build_adjacency(X, GRM_df, gcfg_final, node_idx=None)
            edge_index, edge_weight, _ = to_sparse(A_all, device)

            # FS based on OUTER-TRAIN only; apply columns to all nodes for forward pass
            cols = slice(None)
            if bool(best.get("use_snp_selection", False)):
                k = int(best.get("num_snps", X.shape[1]))
                cols = _select_top_snps_by_abs_corr(X[idx_outer_train], y[idx_outer_train], min(k, X.shape[1]))

            x_all = torch.from_numpy(X[:, cols]).to(device)
            y_all = torch.from_numpy(y).to(device).float()

            model = make_model(in_dim=x_all.shape[1], tp=tp_final).to(device)
            opt = _optimizer(tp_final.optimizer, model.parameters(), tp_final.lr, tp_final.weight_decay)
            loss_fn = make_loss(tp_final.loss_name)

            # Final train uses all OUTER-TRAIN nodes (no inner masking now), OUTER-TEST stays masked.
            train_masked_epochs(model, x_all, edge_index, edge_weight, y_all,
                                train_idx=idx_outer_train, epochs=tp_final.epochs, opt=opt, loss_fn=loss_fn)
            model.eval()
            with torch.no_grad():
                yhat_all = model(x_all, edge_index, edge_weight).detach().cpu().numpy().ravel()
            r_test = _pearson_corr(y_eval[idx_outer_test], yhat_all[idx_outer_test])

        else:  # INDUCTIVE final: train on OUTER-TRAIN graph, eval on OUTER-TEST graph
            # FS refit on OUTER-TRAIN only if enabled (per best params)
            cols = slice(None)
            if bool(best.get("use_snp_selection", False)):
                k = int(best.get("num_snps", X.shape[1]))
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
            opt = _optimizer(tp_final.optimizer, model.parameters(), tp_final.lr, tp_final.weight_decay)
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
    if selected_set:
        # append suffix to indicate which outer splits were run
        suffix = "splits_" + "_".join(str(i) for i in sorted(selected_set))
        out_name = f"{out_name}_{suffix}"
    os.makedirs(out_dir, exist_ok=True)
    summary = {
        "mode": learning_mode,
        "cv_strategy": strategy,
        "outer_test_corr": outer_results,
        "outer_test_corr_mean": float(np.mean(outer_results)) if outer_results else None,
        "outer_test_corr_std": float(np.std(outer_results)) if outer_results else None,
        "outer_test_corr_weighted": outer_results_weighted if outer_results_weighted else None,
        "outer_test_corr_weighted_mean": float(np.mean(outer_results_weighted)) if outer_results_weighted else None,
        "outer_test_corr_weighted_std": float(np.std(outer_results_weighted)) if outer_results_weighted else None,
        "inner_splits": inner_splits,
    "outer_splits": int(len(selected_set)) if selected_set else outer_splits,
    "selected_splits": sorted(selected_set) if selected_set else None,
        "best_params_per_fold": best_params_per_fold,
    }
    with open(os.path.join(out_dir, f"{out_name}_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    mean_r = summary["outer_test_corr_mean"]
    std_r = summary["outer_test_corr_std"]
    if mean_r is not None and std_r is not None:
        logger.info(f"DONE. Mean OUTER r = {mean_r:.4f} ± {std_r:.4f}")
    else:
        logger.info("DONE. No outer folds were evaluated or results are empty.")


# ------------------------------ CLI ------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Unified nested CV (inner folds) with transductive/inductive control + PyG GCN")
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument(
        "--selected_splits",
        type=str,
        default=None,
        help="Optional: JSON list or comma-separated 1-based outer split indices to run (e.g., '[10,11]' or '10,11'). Use 'false' to disable.",
    )
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    # CLI override for selected_splits if provided
    if args.selected_splits is not None:
        s = args.selected_splits.strip()
        if s.lower() in ("false", "none", "", "0"):
            pass
        else:
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    cfg.setdefault("cv", {})["selected_splits"] = [int(x) for x in parsed]
                else:
                    # fallback to comma-separated
                    vals = [int(x) for x in s.split(",") if x.strip()]
                    cfg.setdefault("cv", {})["selected_splits"] = vals
            except Exception:
                try:
                    vals = [int(x) for x in s.split(",") if x.strip()]
                    cfg.setdefault("cv", {})["selected_splits"] = vals
                except Exception:
                    raise ValueError("--selected_splits must be a JSON list or comma-separated integers, or 'false'.")
    run_nested_cv(cfg)


if __name__ == "__main__":
    main()
