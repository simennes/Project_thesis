# src/visualize.py
from __future__ import annotations

import argparse
import json
import os
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np


def load_fold_corrs(path: str) -> List[float]:
    with open(path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    per_fold = blob.get("per_fold", [])
    if not per_fold:
        raise ValueError(f"No 'per_fold' key found in {path}.")
    corrs = [float(item["pearson_r"]) for item in per_fold if "pearson_r" in item]
    if not corrs:
        raise ValueError("No 'pearson_r' values found in 'per_fold'.")
    return corrs


def boxplot_corrs_multi(data: Dict[str, List[float]], title: str = "10-fold OOF Pearson r") -> None:
    """Plot multiple boxplots side by side for phenotype -> per-fold correlations."""
    labels = list(data.keys())
    series = [data[k] for k in labels]
    n = len(series)
    fig_w = max(6, int(1.8 * n))
    fig, ax = plt.subplots(figsize=(fig_w, 4))
    bp = ax.boxplot(
        series,
        vert=True,
        showmeans=True,
        meanline=True,
        patch_artist=True,
        widths=0.6,
    )
    # Color palette
    colors = plt.cm.tab10.colors if n <= 10 else plt.cm.tab20.colors
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.4)
    # jittered scatter of fold points for visibility
    for i, corrs in enumerate(series, start=1):
        x = np.random.normal(loc=i, scale=0.04, size=len(corrs))
        ax.scatter(x, corrs, alpha=0.8, s=12, color=colors[(i-1) % len(colors)])

    ax.set_title(title)
    ax.set_ylabel("Pearson correlation (r)")
    ax.set_xticks(list(range(1, n + 1)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()


def discover_cv_runs(root_dir: str) -> Dict[str, str]:
    """Return mapping phenotype -> path_to_cv_metrics.json for subfolders in root_dir."""
    mapping: Dict[str, str] = {}
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    for name in sorted(os.listdir(root_dir)):
        sub = os.path.join(root_dir, name)
        if not os.path.isdir(sub):
            continue
        metrics_path = os.path.join(sub, "cv_metrics.json")
        if os.path.isfile(metrics_path):
            mapping[name] = metrics_path
    if not mapping:
        raise FileNotFoundError(f"No phenotype subfolders with cv_metrics.json found under {root_dir}")
    return mapping


def main() -> None:
    ap = argparse.ArgumentParser(description="Box plots of 10-fold OOF correlations from cv_metrics.json")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--metrics",
        type=str,
        help="Path to a single cv_metrics.json (produced by cross_validate.py)",
    )
    src.add_argument(
        "--dir",
        type=str,
        help="Path to a directory containing phenotype subfolders, each with cv_metrics.json",
    )
    ap.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save the figure (e.g., outputs/cv_runs/boxplot_oof.png or .pdf)",
    )
    ap.add_argument(
        "--title",
        type=str,
        default="10-fold OOF Pearson r",
        help="Plot title",
    )
    args = ap.parse_args()

    if args.metrics:
        # Single file mode (backward compatible)
        corrs = load_fold_corrs(args.metrics)
        data = {"OOF r": corrs}
    else:
        # Directory mode: discover phenotype subfolders
        mapping = discover_cv_runs(args.dir)
        data = {name: load_fold_corrs(path) for name, path in mapping.items()}

    # Print quick stats
    for name, corrs in data.items():
        arr = np.array(corrs, dtype=float)
        print(f"[{name}] Per-fold Pearson r:", [round(float(c), 4) for c in arr.tolist()])
        print(
            f"[{name}] Summary | mean={arr.mean():.4f}, median={np.median(arr):.4f}, std={arr.std(ddof=1):.4f}, min={arr.min():.4f}, max={arr.max():.4f}"
        )

    boxplot_corrs_multi(data, title=args.title)

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        plt.savefig(args.save, dpi=200)
        print(f"Saved figure to: {args.save}")

    plt.show()


if __name__ == "__main__":
    main()
