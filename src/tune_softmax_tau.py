from __future__ import annotations
import argparse
import json
import os
import gc
from pathlib import Path
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import scipy.sparse as sp

from src.data import load_data
from src.graph import build_adjacency
from src.models import TrainParams, make_model
from src.utils import _pearson_corr, _optimizer, _select_top_snps_by_abs_corr

# Island name mapping (same as in nested_cv.py)
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
NAME_TO_ISLAND_ID: Dict[str, int] = {v: k for k, v in ISLAND_ID_TO_NAME.items()}

# ---------------------------- logging ----------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _make_gcfg(best: Dict[str, Any], gspace: Dict[str, Any]) -> Dict[str, Any]:
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
    return gcfg_final


def _csr_to_edge_index(A: sp.csr_matrix, device: torch.device):
    coo = A.tocoo()
    ei = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long, device=device)
    ew = torch.tensor(coo.data, dtype=torch.float32, device=device)
    return ei, ew


def _train_island_model(X: np.ndarray, y: np.ndarray, GRM_df, idx: np.ndarray,
                        best: Dict[str, Any], gspace: Dict[str, Any], device: torch.device,
                        y_eval: np.ndarray) -> Tuple[Any, slice | np.ndarray, Dict[str, Any]]:
    # Graph config
    gcfg = _make_gcfg(best, gspace)
    # Feature selection columns based on this island
    cols: slice | np.ndarray = slice(None)
    if bool(best.get("use_snp_selection", False)):
        k = int(best.get("num_snps", X.shape[1]))
        cols = _select_top_snps_by_abs_corr(X[idx], y[idx], min(k, X.shape[1]))

    # Build train graph and tensors
    A_tr = build_adjacency(X, GRM_df, gcfg, node_idx=idx)
    ei_tr, ew_tr = _csr_to_edge_index(A_tr, device)
    x_tr = torch.from_numpy(X[idx][:, cols]).to(device)
    y_tr_t = torch.from_numpy(y[idx]).to(device).float()

    tp = TrainParams(
        lr=best.get("lr"), weight_decay=best.get("weight_decay"), epochs=best.get("epochs"),
        loss_name=best.get("loss"), optimizer=best.get("optimizer"),
        hidden_dims=best.get("hidden_dims"), dropout=best.get("dropout"),
        batch_norm=bool(best.get("batch_norm")),
        model_type=best.get("model_type", "gcn"),
        gat_heads=best.get("gat_heads", None), gat_attn_dropout=best.get("gat_attn_dropout", None),
        gat_concat_hidden=best.get("gat_concat_hidden", None),
    )
    model = make_model(in_dim=x_tr.shape[1], tp=tp).to(device)
    opt = _optimizer(tp.optimizer, model.parameters(), tp.lr, tp.weight_decay)
    loss_fn = torch.nn.L1Loss() if (tp.loss_name or "mse").lower() == "mae" else torch.nn.MSELoss()

    for _ in range(int(tp.epochs)):
        model.train()
        opt.zero_grad()
        pred = model(x_tr, ei_tr, ew_tr)
        loss = loss_fn(pred, y_tr_t)
        loss.backward()
        opt.step()

    model.eval()
    return model, cols, gcfg


def _predict_on_indices(model, X: np.ndarray, idx: np.ndarray, cols, gcfg: Dict[str, Any], GRM_df, device: torch.device) -> np.ndarray:
    A = build_adjacency(X, GRM_df, gcfg, node_idx=idx)
    ei, ew = _csr_to_edge_index(A, device)
    x = torch.from_numpy(X[idx][:, cols]).to(device)
    with torch.no_grad():
        yhat = model(x, ei, ew).detach().cpu().numpy().ravel()
    return yhat


def _mean_grm_between(GRM_df, idx_a: np.ndarray, idx_b: np.ndarray) -> float:
    if GRM_df is None or idx_a.size == 0 or idx_b.size == 0:
        return float("nan")
    sub = GRM_df.values[np.ix_(idx_a, idx_b)]
    if sub.size == 0:
        return float("nan")
    return float(np.nanmean(sub))


def _softmax_weights(means: np.ndarray, tau: float) -> np.ndarray:
    w = np.array(means, dtype=float)
    if not np.isfinite(w).any():
        return np.ones_like(w) / max(1, len(w))
    finite_vals = w[np.isfinite(w)]
    fill_val = finite_vals.min() if finite_vals.size else 0.0
    w = np.where(np.isfinite(w), w, fill_val)
    z = (w - np.max(w)) / max(tau, 1e-8)
    e = np.exp(z)
    s = e.sum()
    return e / s if s != 0 else np.full_like(w, 1.0 / len(w))


def main():
    ap = argparse.ArgumentParser(description="Tune softmax temperature for inductive ensemble using per-island params from results JSON.")
    ap.add_argument("--config", required=True, type=str, help="Training config JSON used to load data (same as nested CV uses).")
    ap.add_argument("--results", required=True, type=str, help="Results JSON from inductive_ensemble run (contains ensemble_per_island per fold).")
    ap.add_argument("--out", type=str, default=None, help="Output JSON to write tuning summary to.")
    ap.add_argument("--grid", type=str, default="logspace:1e-4:1e-1:40", help="Tau grid spec. Either 'logspace:min:max:n' or comma-separated list.")
    args = ap.parse_args()

    # Tau grid
    if args.grid.startswith("logspace:"):
        _, a, b, n = args.grid.split(":", 3)
        taus = np.logspace(np.log10(float(a)), np.log10(float(b)), int(n)).astype(float)
    else:
        taus = np.array([float(x) for x in args.grid.split(",") if x.strip()], dtype=float)
    logger.info("Tau grid: %d values (%s)", len(taus), args.grid)

    # Load config and data (match nested_cv expectations)
    logger.info("Loading config: %s", args.config)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    base = cfg["base_train"]
    logger.info("Loading data via src.data.load_data …")
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
    logger.info("Data loaded: X=%s, y=%s, islands=%d", X.shape, y.shape, len(np.unique(locality)))

    # Load results JSON (per-fold, per-island params)
    logger.info("Reading ensemble results: %s", args.results)
    with open(args.results, "r", encoding="utf-8") as f:
        results = json.load(f)

    gspace = cfg.get("search_space", {}).get("graph", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # code_to_label coming from loader maps encoded_code -> original_island_code (e.g., 0 -> 20).
    # Build helpers:
    enc_to_orig: Dict[int, int] = {int(k): int(v) for k, v in (code_to_label or {}).items()}
    orig_to_enc: Dict[str, int] = {str(v): int(k) for k, v in enc_to_orig.items()}

    def encoded_to_name(enc_code: int) -> str:
        """Convert encoded locality value to human-readable island name if possible."""
        orig = enc_to_orig.get(int(enc_code))
        if orig is None:
            return str(enc_code)
        return ISLAND_ID_TO_NAME.get(int(orig), str(orig))

    folds = results.get("best_params_per_fold", [])
    logger.info("Found %d folds with per-island parameters", len(folds))
    out_records: List[Dict[str, Any]] = []
    outer_r = []

    for fold_idx, fold_rec in enumerate(folds, start=1):
        ens = fold_rec.get("ensemble_per_island", [])
        if not ens:
            # skip non-ensemble folds
            continue
        # Identify inner/train islands by human-readable label; test island is the one present but not listed
        inner_labels = [str(e.get("island_label")) for e in ens]
        present_codes = set(int(c) for c in np.unique(locality))
        test_label = None
        for c in sorted(present_codes):
            name_c = encoded_to_name(int(c))
            if name_c not in inner_labels:
                test_label = name_c
                break
        if test_label is None:
            # could not determine test island label; skip this fold
            continue
        logger.info("Fold %d/%d: inner islands=%s | test=%s", fold_idx, len(folds), ", ".join(inner_labels), test_label)

        # Map labels -> sample indices
        def idx_for_label(lbl: str) -> np.ndarray:
            # 1) If given a human-readable name, map to original code then encoded code
            if lbl in NAME_TO_ISLAND_ID:
                orig = NAME_TO_ISLAND_ID[lbl]
                enc = orig_to_enc.get(str(orig))
                if enc is None:
                    # fall back to original code if locality already stores original codes
                    enc = orig
                return np.where(locality.astype(int) == int(enc))[0]
            # 2) If a numeric string, try as original code -> encoded; else treat as encoded
            try:
                val = int(lbl)
                enc = orig_to_enc.get(str(val), None)
                if enc is None:
                    enc = val
                return np.where(locality.astype(int) == int(enc))[0]
            except Exception:
                return np.array([], dtype=int)

        inner_island_indices = {lbl: idx_for_label(lbl) for lbl in inner_labels}
        test_idx = idx_for_label(test_label)

        # Train per-island models and collect predictions
        models = {}
        model_meta = {}
        for e in ens:
            lbl = str(e["island_label"])
            idx = inner_island_indices[lbl]
            best = e.get("best_params", {})
            logger.info("  Training model for island '%s' (n=%d)", lbl, idx.size)
            # Train model on its island
            model, cols, gcfg = _train_island_model(X, y, GRM_df, idx, best, gspace, device, y_eval)
            models[lbl] = model
            model_meta[lbl] = {"cols": cols, "gcfg": gcfg, "idx_train": idx}

        # For each target inner island k: get predictions from other models and tune tau
        best_taus = []
        for tgt_lbl, tgt_idx in inner_island_indices.items():
            logger.info("  Tuning tau for target island '%s' using %d peer models (grid=%d)", tgt_lbl, len(models) - 1, len(taus))
            preds = []
            means = []
            for src_lbl, model in models.items():
                if src_lbl == tgt_lbl:
                    continue
                meta = model_meta[src_lbl]
                yhat = _predict_on_indices(model, X, tgt_idx, meta["cols"], meta["gcfg"], GRM_df, device)
                preds.append(yhat)
                m = _mean_grm_between(GRM_df, tgt_idx, meta["idx_train"])
                means.append(m)
            if not preds:
                continue
            Y = np.vstack(preds)  # (n_models-1, n_tgt)
            y_true = y_eval[tgt_idx]
            # grid search tau
            best_r = -1.0
            best_tau = float(taus[0])
            for tau in taus:
                w = _softmax_weights(np.array(means, dtype=float), float(tau))
                yhat = np.sum((w[:, None]) * Y, axis=0)
                r = _pearson_corr(y_true, yhat)
                if r > best_r:
                    best_r = r
                    best_tau = float(tau)
            best_taus.append(best_tau)
            logger.info("    Best tau for '%s': %.6f (r=%.4f)", tgt_lbl, best_tau, best_r)

        if not best_taus:
            continue
        tau_avg = float(np.mean(best_taus))
        logger.info("  Fold %d: tau_avg over %d islands = %.6f", fold_idx, len(best_taus), tau_avg)

        # Evaluate on true test island with averaged tau
        preds_test = []
        means_test = []
        for src_lbl, model in models.items():
            meta = model_meta[src_lbl]
            yhat_te = _predict_on_indices(model, X, test_idx, meta["cols"], meta["gcfg"], GRM_df, device)
            preds_test.append(yhat_te)
            m_te = _mean_grm_between(GRM_df, test_idx, meta["idx_train"])
            means_test.append(m_te)
        Y_te = np.vstack(preds_test)
        w_te = _softmax_weights(np.array(means_test, dtype=float), tau_avg)
        yhat_te = np.sum((w_te[:, None]) * Y_te, axis=0)
        r_test = _pearson_corr(y_eval[test_idx], yhat_te)
        outer_r.append(float(r_test))
        logger.info("  Fold %d: test '%s' r=%.4f", fold_idx, test_label, r_test)

        out_records.append({
            "fold": int(fold_rec.get("fold", -1)),
            "test_island_label": test_label,
            "tau_inner_best": best_taus,
            "tau_avg": tau_avg,
            "r_test": float(r_test),
        })

        # cleanup models
        for m in models.values():
            del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    out = {
        "source_results": str(args.results),
        "grid": taus.tolist(),
        "per_fold": out_records,
        "outer_mean_r": float(np.mean(outer_r)) if outer_r else None,
        "outer_std_r": float(np.std(outer_r)) if outer_r else None,
    }

    out_path = args.out
    if not out_path:
        stem = Path(args.results).stem
        out_path = str(Path(args.results).with_name(f"tau_tuning_{stem}.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info("Saved tau tuning summary to: %s", out_path)
    if out["outer_mean_r"] is not None:
        logger.info("Overall mean r = %.4f ± %.4f", out["outer_mean_r"], out["outer_std_r"]) 


if __name__ == "__main__":
    main()
