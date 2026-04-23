"""
Temporal EF Model – Standalone Visualizer
==========================================
Generates all diagnostic plots for the Temporal CNN+LSTM model.
Reads  : temporal_epoch_log.csv  (in this folder)
Outputs: training/plots/temporal/

Usage:
    python -m training.models.temporal.temporal_visualizer
"""

import os
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# ── Paths ──────────────────────────────────────────────────────────────────

_HERE        = os.path.dirname(__file__)
_PROJ_ROOT   = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
LOG_PATH     = os.path.join(_HERE, "temporal_epoch_log.csv")
MODEL_PATH   = os.path.join(_HERE, "temporal_model.pt")
OUT_DIR      = os.path.join(_PROJ_ROOT, "training", "plots", "temporal")

# ── Style ──────────────────────────────────────────────────────────────────

PALETTE = {
    "temporal": "#9B59B6",   # purple
    "train"   : "#2ECC71",
    "val"     : "#E74C3C",
    "ref"     : "#95A5A6",
}
DPI = 130
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def _savefig(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Log loading ────────────────────────────────────────────────────────────

def load_log(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


# ── Plot helpers ───────────────────────────────────────────────────────────

def plot_training_dynamics(log):
    """01 – Loss + MAE curves over epochs."""
    epochs = [r["epoch"] for r in log]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Temporal Model – Training Dynamics", fontweight="bold")

    # Loss
    ax = axes[0]
    ax.plot(epochs, [r["train_loss"] for r in log], color=PALETTE["train"], label="Train Loss")
    ax.plot(epochs, [r["val_loss"]   for r in log], color=PALETTE["val"],   label="Val Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Huber Loss")
    ax.set_title("Loss"); ax.legend()

    # MAE
    ax = axes[1]
    ax.plot(epochs, [r["train_MAE"] for r in log], color=PALETTE["train"], label="Train MAE")
    ax.plot(epochs, [r["val_MAE"]   for r in log], color=PALETTE["val"],   label="Val MAE")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MAE (%EF)")
    ax.set_title("MAE"); ax.legend()

    _savefig(fig, "01_training_dynamics.png")


def plot_pred_vs_actual(preds, targets):
    """02 – Scatter: predicted vs actual EF."""
    lo = min(targets.min(), preds.min()) - 2
    hi = max(targets.max(), preds.max()) + 2

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(targets, preds, alpha=0.35, color=PALETTE["temporal"], s=8, label="Test samples")
    ax.plot([lo, hi], [lo, hi], "--", color=PALETTE["ref"], linewidth=1.2, label="Identity")
    ax.set_xlabel("Actual EF (%)"); ax.set_ylabel("Predicted EF (%)")
    ax.set_title("Temporal Model – Predicted vs Actual EF", fontweight="bold")
    ax.legend(); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

    _savefig(fig, "02_pred_vs_actual.png")


def plot_error_histogram(preds, targets):
    """03 – Distribution of absolute errors."""
    errors = (preds - targets).abs().numpy()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errors, bins=40, color=PALETTE["temporal"], edgecolor="white", alpha=0.85)
    ax.axvline(errors.mean(), color=PALETTE["ref"], linestyle="--",
               label=f"Mean = {errors.mean():.2f}%")
    ax.set_xlabel("Absolute Error (%EF)"); ax.set_ylabel("Count")
    ax.set_title("Temporal Model – Error Distribution", fontweight="bold")
    ax.legend()

    _savefig(fig, "03_error_histogram.png")


def plot_clinical_accuracy(preds, targets):
    """04 – Bar chart of Acc@2, Acc@3, Acc@5."""
    thresholds = [2, 3, 5]
    accs = [(preds - targets).abs().le(t).float().mean().item() * 100 for t in thresholds]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([f"Acc@{t}" for t in thresholds], accs, color=PALETTE["temporal"], width=0.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)"); ax.set_title("Temporal Model – Clinical Accuracy", fontweight="bold")
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    _savefig(fig, "04_clinical_accuracy.png")


def plot_tight_range_curve(preds, targets):
    """05 – Cumulative accuracy vs tolerance threshold."""
    ths = np.linspace(0, 15, 200)
    cumaccs = [(preds - targets).abs().le(t).float().mean().item() * 100 for t in ths]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ths, cumaccs, color=PALETTE["temporal"], linewidth=1.8)
    for t in (2, 3, 5):
        v = (preds - targets).abs().le(t).float().mean().item() * 100
        ax.axvline(t, linestyle=":", color=PALETTE["ref"], linewidth=1)
        ax.text(t + 0.15, 10, f"Acc@{t}={v:.1f}%", fontsize=8, color=PALETTE["ref"])
    ax.set_xlabel("Error Tolerance (%EF)"); ax.set_ylabel("Cumulative Accuracy (%)")
    ax.set_title("Temporal Model – Tight Range Curve", fontweight="bold")

    _savefig(fig, "05_tight_range_curve.png")


def plot_residual_analysis(preds, targets):
    """06 – Residuals vs actual EF."""
    residuals = (preds - targets).numpy()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(targets.numpy(), residuals, alpha=0.35, color=PALETTE["temporal"], s=8)
    ax.axhline(0, color=PALETTE["ref"], linestyle="--", linewidth=1.2)
    ax.set_xlabel("Actual EF (%)"); ax.set_ylabel("Residual (pred − true)")
    ax.set_title("Temporal Model – Residual Analysis", fontweight="bold")

    _savefig(fig, "06_residual_analysis.png")


def plot_ef_range_performance(preds, targets):
    """07 – MAE broken down by EF quintile."""
    tvals = targets.numpy()
    pvals = preds.numpy()
    quintiles = np.percentile(tvals, [0, 20, 40, 60, 80, 100])
    labels, maes = [], []

    for i in range(len(quintiles) - 1):
        lo, hi = quintiles[i], quintiles[i + 1]
        mask = (tvals >= lo) & (tvals < hi) if i < 4 else (tvals >= lo) & (tvals <= hi)
        if mask.sum() == 0:
            continue
        labels.append(f"{lo:.0f}–{hi:.0f}%")
        maes.append(abs(pvals[mask] - tvals[mask]).mean())

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, maes, color=PALETTE["temporal"])
    ax.set_xlabel("EF Range (%)"); ax.set_ylabel("MAE (%EF)")
    ax.set_title("Temporal Model – MAE by EF Range", fontweight="bold")

    _savefig(fig, "07_ef_range_performance.png")


def plot_hard_cases(preds, targets, top_n=20):
    """08 – Highlight the worst-predicted samples."""
    errors = (preds - targets).abs().numpy()
    idx    = np.argsort(errors)[-top_n:][::-1]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(top_n), errors[idx], color=PALETTE["temporal"])
    ax.set_xlabel("Sample rank (worst → least)"); ax.set_ylabel("Absolute Error (%EF)")
    ax.set_title(f"Temporal Model – Top {top_n} Hard Cases", fontweight="bold")

    _savefig(fig, "08_hard_cases.png")


def plot_epoch_variance(log):
    """09 – Rolling std of val MAE (training stability)."""
    val_mae = np.array([r["val_MAE"] for r in log])
    window  = min(5, len(val_mae))
    rolling_std = [val_mae[max(0, i - window + 1): i + 1].std() for i in range(len(val_mae))]
    epochs  = [r["epoch"] for r in log]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, rolling_std, color=PALETTE["temporal"], linewidth=1.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel(f"Rolling Std (window={window})")
    ax.set_title("Temporal Model – Training Stability (Val MAE Variance)", fontweight="bold")

    _savefig(fig, "09_epoch_variance.png")


def plot_ef_distribution(targets):
    """10 – EF distribution of the test set."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(targets.numpy(), bins=40, color=PALETTE["temporal"], edgecolor="white", alpha=0.85)
    ax.set_xlabel("EF (%)"); ax.set_ylabel("Count")
    ax.set_title("Temporal Model – Test EF Distribution", fontweight="bold")

    _savefig(fig, "10_ef_distribution.png")


# ── Inference on test set ─────────────────────────────────────────────────

def get_test_predictions():
    from training.temporal_dataset_loader import TemporalEchoDataset
    from training.models.temporal.temporal_model import TemporalEFModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = TemporalEFModel().to(device)
    state  = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    test_ds     = TemporalEchoDataset(split=2)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            preds     = model(sequences).squeeze(-1).cpu()
            all_preds.append(preds)
            all_targets.append(labels)

    return torch.cat(all_preds), torch.cat(all_targets)


# ── Entry point ───────────────────────────────────────────────────────────

def run_all():
    print(f"\n{'='*55}")
    print(f"  Temporal Model Visualizer")
    print(f"  Output directory: {OUT_DIR}")
    print(f"{'='*55}\n")

    # Epoch log plots
    if not os.path.exists(LOG_PATH):
        print(f"  [ERROR] Log not found: {LOG_PATH}\n  Run the trainer first.")
        return

    log = load_log(LOG_PATH)
    print("[1] Training dynamics...")
    plot_training_dynamics(log)
    print("[2] Epoch variance...")
    plot_epoch_variance(log)

    # Test-set inference plots
    if not os.path.exists(MODEL_PATH):
        print(f"\n  [ERROR] Checkpoint not found: {MODEL_PATH}\n  Run the trainer first.")
        return

    print("[3] Running inference on test set...")
    preds, targets = get_test_predictions()
    print(f"     Test samples: {len(preds)}")

    print("[4] Pred vs Actual...")
    plot_pred_vs_actual(preds, targets)
    print("[5] Error Histogram...")
    plot_error_histogram(preds, targets)
    print("[6] Clinical Accuracy...")
    plot_clinical_accuracy(preds, targets)
    print("[7] Tight Range Curve...")
    plot_tight_range_curve(preds, targets)
    print("[8] Residual Analysis...")
    plot_residual_analysis(preds, targets)
    print("[9] EF Range Performance...")
    plot_ef_range_performance(preds, targets)
    print("[10] Hard Cases...")
    plot_hard_cases(preds, targets)
    print("[11] EF Distribution...")
    plot_ef_distribution(targets)

    print(f"\n  All plots saved to: {OUT_DIR}\n")


if __name__ == "__main__":
    run_all()
