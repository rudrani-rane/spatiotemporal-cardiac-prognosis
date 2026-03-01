"""
Ensemble Runner
===============
Evaluates the ensemble (MotionDifferenceModel + HybridEFModel averaged)
on all three splits: train, val, test.

Saves per-split metrics to ensemble_metrics.csv in this directory so
the visualizer can read and plot them.

Usage:
    python -m training.models.ensemble.ensemble_runner
"""

import os
import csv
import torch
import numpy as np

from training.models.ensemble.ensemble_predictor import EnsemblePredictor
from training.metrics import compute_all_metrics

# ── Paths ──────────────────────────────────────────────────────────────────

SAVE_DIR = os.path.dirname(__file__)
LOG_PATH = os.path.join(SAVE_DIR, "ensemble_metrics.csv")

# ── CSV schema ─────────────────────────────────────────────────────────────

CSV_HEADER = [
    "split",
    "MAE", "RMSE", "MAPE", "R2",
    "Acc@5", "Acc@3", "Acc@2",
]


def _write_row(path, split_name, metrics):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([
            split_name,
            round(metrics["MAE"],   4),
            round(metrics["RMSE"],  4),
            round(metrics["MAPE"],  4),
            round(metrics["R2"],    4),
            round(metrics["Acc@5"], 4),
            round(metrics["Acc@3"], 4),
            round(metrics["Acc@2"], 4),
        ])


# ── Main ───────────────────────────────────────────────────────────────────

def run_evaluation():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  Ensemble Predictor – Evaluation")
    print(f"  Device  : {device}")
    print(f"{'='*55}\n")

    predictor = EnsemblePredictor(device=device)

    # ── Find optimal weights on val split ─────────────────────────────────
    print("Searching for optimal weights on val split (200 steps)…")
    m_w, h_w, best_val_mae = predictor.find_optimal_weights(val_steps=200)
    print(f"  Motion Diff weight : {m_w:.4f}")
    print(f"  Hybrid EF   weight : {h_w:.4f}")
    print(f"  Best Val MAE       : {best_val_mae:.4f}\n")

    # Initialise CSV
    with open(LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow(CSV_HEADER)

    split_map = {0: "train", 1: "val", 2: "test"}

    for split_id, split_name in split_map.items():
        print(f"Evaluating split : {split_name}  (split={split_id})")
        ens_preds, true_labels, motion_preds, hybrid_preds = predictor.predict(
            split=split_id
        )

        preds_t  = torch.tensor(ens_preds)
        labels_t = torch.tensor(true_labels)
        metrics  = compute_all_metrics(preds_t, labels_t)

        print(f"  Samples : {len(true_labels)}")
        print(f"  {'Metric':<10} {'Value':>10}")
        print(f"  {'-'*22}")
        print(f"  {'MAE':<10} {metrics['MAE']:>10.3f}")
        print(f"  {'RMSE':<10} {metrics['RMSE']:>10.3f}")
        print(f"  {'MAPE':<10} {metrics['MAPE']:>10.3f}")
        print(f"  {'R²':<10} {metrics['R2']:>10.4f}")
        print(f"  {'Acc@5':<10} {metrics['Acc@5']*100:>9.2f}%")
        print(f"  {'Acc@3':<10} {metrics['Acc@3']*100:>9.2f}%")
        print(f"  {'Acc@2':<10} {metrics['Acc@2']*100:>9.2f}%\n")

        _write_row(LOG_PATH, split_name, metrics)

    print(f"{'='*55}")
    print(f"  Metrics saved : {LOG_PATH}")
    print(f"  Weights used  : Motion={predictor.motion_weight:.4f}  Hybrid={predictor.hybrid_weight:.4f}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    run_evaluation()
