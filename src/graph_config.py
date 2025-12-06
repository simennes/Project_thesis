from __future__ import annotations
from typing import Any, Dict


def graph_cfg_from_params(params: Dict[str, Any], graph_space: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Normalize graph-related parameters to a consistent GRM-based config dict."""
    gspace = graph_space or {}
    mode_default = gspace.get("graph_mode_default", "knn")
    graph_mode = str(params.get("graph_mode", mode_default)).lower()
    cfg: Dict[str, Any] = {"graph_mode": graph_mode}
    if graph_mode == "off":
        return cfg

    self_loops_default = bool(gspace.get("self_loops_default", False))

    if graph_mode == "knn":
        cfg.update({
            "knn_k": int(params.get("knn_k", gspace.get("knn_k_default", 5))),
            "weighted_edges": bool(params.get("weighted_edges", gspace.get("weighted_edges_default", False))),
            "symmetrize_mode": params.get("symmetrize_mode", gspace.get("symmetrize_mode_default", "union")),
            "self_loops": bool(params.get("self_loops", self_loops_default)),
        })
        return cfg

    if graph_mode == "cutoff":
        cfg.update({
            "cutoff": float(params.get("cutoff", gspace.get("cutoff_default", 0.5))),
            "grm_norm": params.get("grm_norm", gspace.get("grm_norm_default", "none")),
            "self_loops": bool(params.get("self_loops", self_loops_default)),
        })
        return cfg

    raise ValueError(f"Unsupported graph_mode '{graph_mode}'. Expected 'off', 'knn', or 'cutoff'.")
