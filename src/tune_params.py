# src/tune_params.py
import os
import json
import argparse
import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.optim import Adam, SGD

from data import load_data
from graph import build_knn_from_grm, gcn_normalize, sample_subgraph_indices
from gcn import GCN
from utils import set_seed, to_torch_sparse, metrics


def objective(trial, base_cfg, search):
    cfg = json.loads(json.dumps(base_cfg))  # deep copy

    # Sample hyperparams
    model = cfg["model"]
    graph = cfg["graph"]
    train = cfg["training"]

    # Model dims
    hidden_choices = search["model"]["hidden_dims_choices"]
    model["hidden_dims"] = trial.suggest_categorical("hidden_dims", hidden_choices)
    model["dropout"] = trial.suggest_float("dropout", *search["model"]["dropout_range"])
    model["batch_norm"] = trial.suggest_categorical("batch_norm", [True, False])

    # Graph
    graph["knn_k"] = trial.suggest_int("knn_k", *search["graph"]["knn_k_range"])
    graph["weighted_edges"] = trial.suggest_categorical("weighted_edges", [False, True])
    graph["subgraph_size_fraction"] = trial.suggest_float("subgraph_size_fraction",
                                                          *search["graph"]["subgraph_fraction_range"])
    graph["num_subgraphs_per_epoch"] = trial.suggest_int("num_subgraphs_per_epoch",
                                                         *search["graph"]["num_subgraphs_range"])

    # Training
    train["lr"] = trial.suggest_float("lr", *search["training"]["lr_loguniform"])
    train["weight_decay"] = trial.suggest_float("weight_decay", *search["training"]["wd_loguniform"])
    train["optimizer"] = trial.suggest_categorical("optimizer", ["adam", "sgd"])
    train["epochs"] = search["training"]["epochs"]
    train["val_fraction"] = base_cfg["training"]["val_fraction"]
    train["test_fraction"] = base_cfg["training"]["test_fraction"]

    seed = base_cfg.get("seed", 42)
    set_seed(seed)

    # Load data once per trial
    X, y, ids, GRM_df = load_data(base_cfg["paths"],
                                  target_column=base_cfg.get("target_column", "y_adjusted"),
                                  standardize_features=base_cfg.get("standardize_features", False))

    # Graph
    A = build_knn_from_grm(GRM_df,
                           k=graph["knn_k"],
                           weighted_edges=graph["weighted_edges"],
                           symmetrize_mode=graph.get("symmetrize_mode", "union"),
                           add_self_loops=graph.get("self_loops", True))
    A_norm = gcn_normalize(A)
    A_idx, A_val, A_shape = to_torch_sparse(A_norm)

    # Splits (fixed)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(n * train["test_fraction"])
    n_val = int(n * train["val_fraction"])
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_t = GCN(X.shape[1], model["hidden_dims"], dropout=model["dropout"], use_bn=model["batch_norm"]).to(device)
    opt = Adam(model_t.parameters(), lr=train["lr"], weight_decay=train["weight_decay"]) if train["optimizer"] == "adam" \
          else SGD(model_t.parameters(), lr=train["lr"], momentum=0.9, weight_decay=train["weight_decay"])
    loss_fn = nn.MSELoss()
    A_idx_dev = A_idx.to(device)
    A_val_dev = A_val.to(device)

    # short training for tuning
    sub_frac = graph["subgraph_size_fraction"]
    subs_per_epoch = graph["num_subgraphs_per_epoch"]
    for epoch in range(train["epochs"]):
        # train
        for _ in range(subs_per_epoch):
            sub = sample_subgraph_indices(train_idx, sub_frac, rng)
            # build induced subgraph tensors
            rows = A_idx[0].numpy(); cols = A_idx[1].numpy()
            mask = np.zeros(A_shape[0], dtype=bool); mask[sub] = True
            keep = mask[rows] & mask[cols]
            remap = -np.ones(A_shape[0], dtype=int); remap[sub] = np.arange(len(sub))
            sub_indices = torch.tensor([remap[rows[keep]], remap[cols[keep]]], dtype=torch.long, device=device)
            sub_values = torch.tensor(A_val.numpy()[keep], dtype=torch.float32, device=device)
            sub_shape = (len(sub), len(sub))

            x_sub = torch.from_numpy(X[sub]).to(device)
            y_sub = torch.from_numpy(y[sub]).to(device)
            model_t.train(); opt.zero_grad()
            pred = model_t(x_sub, sub_indices, sub_values, sub_shape)
            loss = loss_fn(pred, y_sub)
            loss.backward(); opt.step()

        # validate on full graph
        model_t.eval()
        with torch.no_grad():
            preds = model_t(torch.from_numpy(X).to(device), A_idx_dev, A_val_dev, A_shape).cpu().numpy()
            val_m = metrics(y[val_idx], preds[val_idx])
        trial.report(val_m["rmse"], epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_m["rmse"]


def main(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    base_cfg = cfg["base_train"]
    search = cfg["search_space"]

    study = optuna.create_study(direction="minimize",
                                study_name=cfg.get("study_name", "gcn_tuning"),
                                sampler=optuna.samplers.TPESampler(seed=base_cfg.get("seed", 42)),
                                pruner=optuna.pruners.MedianPruner(
                                    n_startup_trials=cfg.get("n_startup_trials", 5),
                                    n_warmup_steps=cfg.get("n_warmup_epochs", 5)
                                ))
    study.optimize(lambda tr: objective(tr, base_cfg, search),
                   n_trials=cfg.get("n_trials", 40),
                   timeout=cfg.get("timeout_seconds", None),
                   gc_after_trial=True)

    os.makedirs(base_cfg["paths"]["output_dir"], exist_ok=True)
    with open(os.path.join(base_cfg["paths"]["output_dir"], "tuning_results.json"), "w") as f:
        json.dump({"best_value": study.best_value,
                   "best_params": study.best_params}, f, indent=2)
    print("Best RMSE:", study.best_value)
    print("Best params:", study.best_params)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to config_tune.json")
    args = ap.parse_args()
    main(args.config)
