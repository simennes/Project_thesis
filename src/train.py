import os
import json
import argparse
import logging
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from src.data import load_data
from src.graph import (
    build_knn_from_grm,
    build_knn_from_snp,
    gcn_normalize,
    induce_subgraph,
    naive_partition,
)
from src.gcn import GCN
from src.utils import set_seed, to_sparse, metrics, save_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def split_indices(
    n: int, val_fraction: float, test_fraction: float, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]
    return train_idx, val_idx, test_idx


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"])
    logging.info(f"Device: {device}")

    logging.info("Loading data...")
    X, y, ids, GRM_df = load_data(
        cfg["paths"],
        target_column=cfg.get("target_column", "y_adjusted"),
        standardize_features=cfg.get("standardize_features", False),
    )

    graph_cfg = cfg.get("graph", {})
    graph_source = graph_cfg.get("source", "grm").lower()
    k = graph_cfg.get("knn_k", 5)
    weighted_edges = graph_cfg.get("weighted_edges", False)
    sym_mode = graph_cfg.get("symmetrize_mode", "union")
    self_loops = graph_cfg.get("self_loops", True)
    lap_smoothing = graph_cfg.get("laplacian_smoothing", True)

    n = X.shape[0]
    train_idx, val_idx, test_idx = split_indices(
        n,
        graph_cfg.get("val_fraction", 0.1),
        graph_cfg.get("test_fraction", 0.1),
        seed=cfg["seed"],
    )

    # Build one global adjacency
    logging.info(f"Building one global graph from {graph_source.upper()}...")
    if graph_source == "grm":
        A_global = build_knn_from_grm(
            GRM_df,
            k=k,
            weighted_edges=weighted_edges,
            symmetrize_mode=sym_mode,
            add_self_loops=self_loops,
        )
        A_global = gcn_normalize(A_global)
    else:
        A_global = build_knn_from_snp(
            X,
            k=k,
            weighted_edges=weighted_edges,
            symmetrize_mode=sym_mode,
            add_self_loops=self_loops,
            laplacian_smoothing=lap_smoothing,
        )
        if not lap_smoothing:
            A_global = gcn_normalize(A_global)

    # Induce subgraphs
    A_train = induce_subgraph(A_global, train_idx)
    A_val = induce_subgraph(A_global, val_idx)
    A_test = induce_subgraph(A_global, test_idx)

    A_train_idx, A_train_val, A_train_shape = to_sparse(A_train, device)
    A_val_idx, A_val_val, A_val_shape = to_sparse(A_val, device)
    A_test_idx, A_test_val, A_test_shape = to_sparse(A_test, device)

    # Early stopping configuration
    patience = int(cfg["training"].get("patience", 0))
    has_val = len(val_idx) > 0

    # Ensemble?
    ensemble_count = int(graph_cfg.get("ensemble_models", 1))
    use_ensemble = ensemble_count > 1
    loss_fn = nn.MSELoss()
    lr = cfg["training"]["lr"]
    weight_decay = cfg["training"].get("weight_decay", 0.0)

    if not use_ensemble:
        # Single model
        model = GCN(
            in_dim=X.shape[1],
            hidden_dims=cfg["model"]["hidden_dims"],
            dropout=cfg["model"]["dropout"],
            use_bn=cfg["model"]["batch_norm"],
        ).to(device)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # Pre-create tensors
        X_train_t = torch.from_numpy(X[train_idx]).to(device)
        y_train_t = torch.from_numpy(y[train_idx]).to(device)
        X_val_t = torch.from_numpy(X[val_idx]).to(device) if has_val else None
        y_val_t = torch.from_numpy(y[val_idx]).to(device) if has_val else None
        best_state = None
        best_val = float("inf")
        no_imp = 0
        for epoch in range(1, cfg["training"]["epochs"] + 1):
            model.train()
            optimizer.zero_grad()
            pred = model(X_train_t, A_train_idx, A_train_val, A_train_shape)
            loss = loss_fn(pred, y_train_t)
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                if has_val:
                    pred_val_t = model(X_val_t, A_val_idx, A_val_val, A_val_shape)
                    val_loss_t = loss_fn(pred_val_t, y_val_t)
                    pred_val_np = pred_val_t.detach().cpu().numpy()
                    val_metrics = metrics(y[val_idx], pred_val_np)
                    logging.info(f"Epoch {epoch:03d} | train_loss={loss.item():.4f} | val_loss={val_loss_t.item():.4f} | VAL metrics: {val_metrics}")
                    cur = float(val_loss_t.item())
                    if patience > 0:
                        if cur < best_val - 1e-8:
                            best_val = cur
                            best_state = copy.deepcopy(model.state_dict())
                            no_imp = 0
                        else:
                            no_imp += 1
                            if no_imp >= patience:
                                logging.info(f"Early stopping at epoch {epoch} (best val_loss={best_val:.6f})")
                                break
                else:
                    logging.info(f"Epoch {epoch:03d} | train_loss={loss.item():.4f}")
        if best_state is not None:
            model.load_state_dict(best_state)
        # Test
        model.eval()
        with torch.no_grad():
            X_te = torch.from_numpy(X[test_idx]).to(device)
            pred_te = model(X_te, A_test_idx, A_test_val, A_test_shape).cpu().numpy()
        test_metrics = metrics(y[test_idx], pred_te)
        logging.info(f"TEST metrics: {test_metrics}")

    else:
        # Partition train graph into ensemble subgraphs
        edge_list = list(zip(*A_train.nonzero()))
        edge_list = [(u, v) for (u, v) in edge_list if u != v]
        traversed = set()
        subgraph_indices = []
        size = max(1, len(train_idx) // ensemble_count)
        logging.info(f"Partitioning training graph into {ensemble_count} subgraphs of size ~{size}...")
        for _ in range(ensemble_count):
            nodes = naive_partition(edge_list, size, traversed=traversed)
            traversed |= set(nodes)
            subgraph_indices.append(nodes)

        # Prepare validation tensors once
        X_val_t = torch.from_numpy(X[val_idx]).to(device) if has_val else None
        y_val_t = torch.from_numpy(y[val_idx]).to(device) if has_val else None

        models = []
        for i, nodes in enumerate(subgraph_indices):
            X_sub = X[train_idx][nodes]
            y_sub = y[train_idx][nodes]
            A_sub = induce_subgraph(A_train, nodes)
            A_sub_idx, A_sub_val, A_sub_shape = to_sparse(A_sub, device)
            X_sub_t = torch.from_numpy(X_sub).to(device)
            y_sub_t = torch.from_numpy(y_sub).to(device)
            model_i = GCN(
                in_dim=X.shape[1],
                hidden_dims=cfg["model"]["hidden_dims"],
                dropout=cfg["model"]["dropout"],
                use_bn=cfg["model"]["batch_norm"],
            ).to(device)
            optimizer_i = Adam(model_i.parameters(), lr=lr, weight_decay=weight_decay)
            best_sub_state = None
            best_sub_val = float("inf")
            no_imp = 0
            for epoch in range(1, cfg["training"]["epochs"] + 1):
                model_i.train()
                optimizer_i.zero_grad()
                pred = model_i(X_sub_t, A_sub_idx, A_sub_val, A_sub_shape)
                loss = loss_fn(pred, y_sub_t)
                loss.backward()
                optimizer_i.step()
                model_i.eval()
                with torch.no_grad():
                    if has_val:
                        pred_val_t = model_i(X_val_t, A_val_idx, A_val_val, A_val_shape)
                        val_loss_t = loss_fn(pred_val_t, y_val_t)
                        pred_val_np = pred_val_t.detach().cpu().numpy()
                        val_metrics = metrics(y[val_idx], pred_val_np)
                        logging.info(
                            f"Model {i+1}/{ensemble_count} Epoch {epoch:03d} | train_loss={loss.item():.4f} | val_loss={val_loss_t.item():.4f} | VAL metrics: {val_metrics}"
                        )
                        cur = float(val_loss_t.item())
                        if patience > 0:
                            if cur < best_sub_val - 1e-8:
                                best_sub_val = cur
                                best_sub_state = copy.deepcopy(model_i.state_dict())
                                no_imp = 0
                            else:
                                no_imp += 1
                                if no_imp >= patience:
                                    logging.info(
                                        f" Early stopping submodel {i+1}/{ensemble_count} at epoch {epoch} (best val_loss={best_sub_val:.6f})"
                                    )
                                    break
                    else:
                        logging.info(
                            f"Model {i+1}/{ensemble_count} Epoch {epoch:03d} | train_loss={loss.item():.4f}"
                        )
            if best_sub_state is not None:
                model_i.load_state_dict(best_sub_state)
            models.append(model_i)

        # Test ensemble: mean predictions
        X_te = torch.from_numpy(X[test_idx]).to(device)
        preds = []
        for m in models:
            m.eval()
            with torch.no_grad():
                preds.append(m(X_te, A_test_idx, A_test_val, A_test_shape))
        preds_stack = torch.stack(preds, dim=1)
        final_pred = preds_stack.mean(dim=1).cpu().numpy()
        test_metrics = metrics(y[test_idx], final_pred)
        logging.info(f"TEST metrics (mean ensemble): {test_metrics}")

    outdir = cfg["paths"]["output_dir"]
    os.makedirs(outdir, exist_ok=True)
    save_json({"config": cfg, "test": test_metrics}, os.path.join(outdir, "summary.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    main(config)
