# src/visualize.py
import argparse
import os
import json
import pandas as pd
import matplotlib.pyplot as plt


def main(run_dir):
    metrics_csv = os.path.join(run_dir, "metrics.csv")
    summary_json = os.path.join(run_dir, "summary.json")
    test_pred_csv = os.path.join(run_dir, "test_predictions.csv")

    hist = pd.read_csv(metrics_csv)
    with open(summary_json, "r") as f:
        summary = json.load(f)
    test = pd.read_csv(test_pred_csv)

    # Curves
    plt.figure()
    plt.plot(hist["epoch"], hist["train_loss"], label="train_loss")
    plt.plot(hist["epoch"], hist["val_rmse"], label="val_rmse")
    plt.xlabel("epoch"); plt.ylabel("loss / RMSE"); plt.legend(); plt.title("Training curves")
    plt.tight_layout(); plt.show()

    # Prediction scatter
    plt.figure()
    plt.scatter(test["y_true"], test["y_pred"], s=12)
    lims = [min(test["y_true"].min(), test["y_pred"].min()),
            max(test["y_true"].max(), test["y_pred"].max())]
    plt.plot(lims, lims)
    plt.xlabel("True"); plt.ylabel("Pred"); plt.title(f"Test (RMSE={summary['test']['rmse']:.3f}, R2={summary['test']['r2']:.3f})")
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    args = ap.parse_args()
    main(args.run_dir)
