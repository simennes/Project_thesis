"""Hyperparameter suggestion utilities for Optuna-based tuning."""
from __future__ import annotations
from typing import Any, Dict
import optuna

from src.models import TrainParams
from src.utils import encode_choices_for_optuna, decode_choice


def suggest_params(trial: optuna.Trial, space: Dict[str, Any]) -> TrainParams:
    """Suggest hyperparameters for a trial based on search space configuration.
    
    Parameters
    ----------
    trial : optuna.Trial
        Current Optuna trial
    space : dict
        Search space configuration with keys: model, training, graph, feature_selection
        
    Returns
    -------
    TrainParams
        Dataclass containing suggested hyperparameters
    """
    m = space.get("model", {})
    t = space.get("training", {})
    g = space.get("graph", {})
    fsel = space.get("feature_selection", {})

    model_type = m.get("model_type_default", "gcn")
    model_type = trial.suggest_categorical("model_type", m.get("model_type_choices", ["gcn", "gat", "graphsage", "mlp"]))

    hidden = m.get("hidden_dims_choices", [])
    hidden = encode_choices_for_optuna(hidden)
    hidden = trial.suggest_categorical("hidden_dims", hidden)

    dropout = trial.suggest_float("dropout", *m.get("dropout_range", (0.0, 0.5)))
    batch_norm = trial.suggest_categorical("batch_norm", m.get("batch_norm_choices", [True, False]))

    # GNN-specific params (conditionally sampled)
    gat_heads = None
    gat_attn_dropout = None
    gat_concat_hidden = None
    sage_aggr = None
    sage_normalize = None
    sage_project = None
    if model_type == "gat":
        gat_heads = trial.suggest_int("gat_heads", *m.get("gat_heads_range", (1, 8)))
        gat_attn_dropout = trial.suggest_float("gat_attn_dropout", *m.get("gat_attn_dropout_range", (0.0, 0.6)))
        gat_concat_hidden = trial.suggest_categorical("gat_concat_hidden", m.get("gat_concat_hidden_choices", [True, False]))
    elif model_type == "graphsage":
        sage_aggr = trial.suggest_categorical("sage_aggr", m.get("sage_aggr_choices", ["mean", "max", "add", "lstm"]))
        sage_normalize = trial.suggest_categorical("sage_normalize", m.get("sage_normalize_choices", [False, True]))
        project_choices = m.get("sage_project_choices", [False, True])
        allowed_project_aggr = {"mean", "max"}
        if sage_aggr in allowed_project_aggr:
            sage_project = trial.suggest_categorical("sage_project", project_choices)
        else:
            # Force project=False for aggregators that explode memory (e.g., lstm/add)
            fallback_choices = [choice for choice in project_choices if choice is False]
            if not fallback_choices:
                fallback_choices = [False]
            sage_project = trial.suggest_categorical("sage_project", fallback_choices)

    lr = trial.suggest_float("lr", *t.get("lr_loguniform", (1e-4, 5e-3)), log=True)
    wd = trial.suggest_float("weight_decay", *t.get("wd_loguniform", (1e-7, 1e-3)), log=True)
    epochs = trial.suggest_int("epochs", *t.get("epochs_range", (50, 300)))
    loss = trial.suggest_categorical("loss", t.get("loss_choices", ["mse", "mae"]))
    opt = trial.suggest_categorical("optimizer", t.get("optimizer_choices", ["adam", "sgd", "adamw"]))

    # Graph hyperparameters suggested by trial so Optuna logs them
    graph_mode = trial.suggest_categorical("graph_mode", g.get("graph_mode_choices", ["knn", "cutoff"]))
    if graph_mode == "knn":
        knn_k = trial.suggest_int("knn_k", *g.get("knn_k_range", (5, 30)))
        weighted_edges = trial.suggest_categorical("weighted_edges", g.get("weighted_edges_choices", [True, False]))
        symmetrize_mode = trial.suggest_categorical("symmetrize_mode", g.get("symmetrize_mode_choices", ["union", "mutual"]))
        self_loops = trial.suggest_categorical("self_loops", g.get("self_loops_choices", [False, True]))
        _ = (knn_k, weighted_edges, symmetrize_mode, self_loops)
    elif graph_mode == "cutoff":
        cutoff = trial.suggest_float("cutoff", *g.get("cutoff_range", (0.2, 0.9)))
        grm_norm = trial.suggest_categorical("grm_norm", g.get("grm_norm_choices", ["none", "cosine", "gcn", "cosine_then_gcn"]))
        self_loops = trial.suggest_categorical("self_loops", g.get("self_loops_choices", [False, True]))
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
                       gat_concat_hidden=gat_concat_hidden,
                       sage_aggr=sage_aggr, sage_normalize=sage_normalize, sage_project=sage_project)
