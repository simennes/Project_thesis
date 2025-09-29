# src/visualize.py
from __future__ import annotations

import argparse
import json
import os
from typing import List

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


def boxplot_corrs(corrs: List[float], title: str = "10-fold OOF Pearson r") -> None:
    # Single boxplot; also overlay individual fold points
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(
        corrs,
        vert=True,
        showmeans=True,
        meanline=True,
        patch_artist=True,
        widths=0.4,
    )
    # jittered scatter of fold points for visibility
    x = np.random.normal(loc=1.0, scale=0.01, size=len(corrs))
    ax.scatter(x, corrs, alpha=0.8)

    ax.set_title(title)
    ax.set_ylabel("Pearson correlation (r)")
    ax.set_xticks([1])
    ax.set_xticklabels(["OOF r"])

    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()


def main() -> None:
    ap = argparse.ArgumentParser(description="Box plot of 10-fold OOF correlations from cv_metrics.json")
    ap.add_argument(
        "--metrics",
        type=str,
        required=True,
        help="Path to cv_metrics.json (produced by cross_validate.py)",
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

    corrs = load_fold_corrs(args.metrics)

    # Print quick stats
    arr = np.array(corrs, dtype=float)
    print("Per-fold Pearson r:", [round(float(c), 4) for c in arr.tolist()])
    print(
        f"Summary | mean={arr.mean():.4f}, median={np.median(arr):.4f}, "
        f"std={arr.std(ddof=1):.4f}, min={arr.min():.4f}, max={arr.max():.4f}"
    )

    boxplot_corrs(corrs, title=args.title)

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        plt.savefig(args.save, dpi=200)
        print(f"Saved figure to: {args.save}")

    plt.show()


if __name__ == "__main__":
    main()
