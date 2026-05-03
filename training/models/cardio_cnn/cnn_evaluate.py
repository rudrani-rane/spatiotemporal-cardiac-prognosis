"""
cnn_evaluate.py
───────────────
Loads the saved CNN checkpoint, runs inference on VAL and TEST splits,
and generates the following plots saved to cardio_cnn/plots/:

  1. true_vs_predicted.png     — scatter: true LVEF vs predicted LVEF
  2. residuals.png             — residual (error) distribution histogram
  3. bland_altman.png          — Bland-Altman agreement plot
  4. error_by_ef_range.png     — MAE broken down by clinical EF category
  5. training_curves.png       — train/val loss + MAE over epochs (from CSV log)

Run from project root:
    python -m training.models.cardio_cnn.cnn_evaluate
"""

import os
import csv
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe on all systems
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from torch.utils.data import DataLoader

from training.models.cardio_cnn.cnn_dataset_loader import CNNEchoDataset
from training.models.cardio_cnn.cnn_model import CardioCNNModel

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
CKPT_PATH   = os.path.join(BASE_DIR, "cnn_model.pth")
LOG_PATH    = os.path.join(BASE_DIR, "cnn_epoch_log.csv")
PLOT_DIR    = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

BATCH_SIZE  = 8
MODEL_LABEL = "Cardio-CNN"

# Clinical EF categories (ACC/AHA guidelines)
EF_BINS   = [0,  40,  50,  55,  70,  100]
EF_LABELS = ["Severely\nReduced\n(<40%)",
             "Mildly\nReduced\n(40–50%)",
             "Low-Normal\n(50–55%)",
             "Normal\n(55–70%)",
             "Hyperdynamic\n(>70%)"]

# ── Colour palette ─────────────────────────────────────────────────────────
C_SCATTER = "#2E75B6"
C_LINE    = "#C55A11"
C_HIST    = "#2E75B6"
C_BAR     = "#2E75B6"
C_BLAND   = "#2E75B6"


# ── Inference ──────────────────────────────────────────────────────────────
def run_inference(split: int, device):
    ds     = CNNEchoDataset(split=split)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = CardioCNNModel().to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device)).squeeze(-1).cpu()
            preds.append(out)
            trues.append(y)

    return torch.cat(trues).numpy(), torch.cat(preds).numpy()


# ── Plot helpers ───────────────────────────────────────────────────────────
def _style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)


# ── Plot 1: True vs Predicted scatter ─────────────────────────────────────
def plot_true_vs_pred(true, pred, split_name):
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(true, pred, alpha=0.45, s=18, color=C_SCATTER, label="Samples")

    lo, hi = min(true.min(), pred.min()) - 2, max(true.max(), pred.max()) + 2
    ax.plot([lo, hi], [lo, hi], color=C_LINE, lw=1.8, ls="--", label="Perfect prediction")

    # ±5% band
    ax.fill_between([lo, hi], [lo-5, hi-5], [lo+5, hi+5],
                    alpha=0.08, color=C_LINE, label="±5 EF band")

    mae  = np.mean(np.abs(true - pred))
    r2   = 1 - np.sum((true - pred)**2) / np.sum((true - np.mean(true))**2)
    corr = np.corrcoef(true, pred)[0, 1]

    ax.text(0.04, 0.96,
            f"MAE  = {mae:.2f}%\nR²   = {r2:.3f}\nr    = {corr:.3f}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#CCCCCC"))

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.legend(fontsize=9, loc="lower right")
    _style_ax(ax,
              f"{MODEL_LABEL} — True vs Predicted LVEF ({split_name})",
              "True LVEF (%)", "Predicted LVEF (%)")
    ax.set_aspect("equal")
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, f"true_vs_predicted_{split_name.lower()}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Plot 2: Residuals histogram ────────────────────────────────────────────
def plot_residuals(true, pred, split_name):
    residuals = pred - true          # positive = over-predicted

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=40, color=C_HIST, edgecolor="white", alpha=0.85)
    ax.axvline(0,  color=C_LINE, lw=2,   ls="--", label="Zero error")
    ax.axvline(residuals.mean(), color="green", lw=1.5, ls=":",
               label=f"Mean = {residuals.mean():.2f}%")

    ax.text(0.97, 0.95,
            f"Std  = {residuals.std():.2f}%\n"
            f"Within ±5: {(np.abs(residuals)<=5).mean()*100:.1f}%\n"
            f"Within ±3: {(np.abs(residuals)<=3).mean()*100:.1f}%",
            transform=ax.transAxes, va="top", ha="right", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#CCCCCC"))

    ax.legend(fontsize=9)
    _style_ax(ax,
              f"{MODEL_LABEL} — Prediction Residuals ({split_name})",
              "Residual: Predicted − True LVEF (%)", "Count")
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, f"residuals_{split_name.lower()}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Plot 3: Bland-Altman ───────────────────────────────────────────────────
def plot_bland_altman(true, pred, split_name):
    mean_vals = (true + pred) / 2
    diff_vals = pred - true

    mean_diff = diff_vals.mean()
    std_diff  = diff_vals.std()
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(mean_vals, diff_vals, alpha=0.4, s=16, color=C_BLAND)
    ax.axhline(mean_diff,  color=C_LINE,  lw=2,   ls="-",
               label=f"Mean bias = {mean_diff:.2f}%")
    ax.axhline(loa_upper,  color="gray",  lw=1.5, ls="--",
               label=f"+1.96 SD = {loa_upper:.2f}%")
    ax.axhline(loa_lower,  color="gray",  lw=1.5, ls="--",
               label=f"−1.96 SD = {loa_lower:.2f}%")
    ax.fill_between(ax.get_xlim(), loa_lower, loa_upper,
                    alpha=0.05, color="gray")

    ax.legend(fontsize=9)
    _style_ax(ax,
              f"{MODEL_LABEL} — Bland-Altman Plot ({split_name})",
              "Mean of True & Predicted LVEF (%)",
              "Difference: Predicted − True LVEF (%)")
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, f"bland_altman_{split_name.lower()}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Plot 4: MAE by clinical EF category ───────────────────────────────────
def plot_error_by_ef_range(true, pred, split_name):
    mae_per_bin = []
    counts      = []
    for lo, hi in zip(EF_BINS[:-1], EF_BINS[1:]):
        mask = (true >= lo) & (true < hi)
        if mask.sum() > 0:
            mae_per_bin.append(np.mean(np.abs(true[mask] - pred[mask])))
            counts.append(mask.sum())
        else:
            mae_per_bin.append(0)
            counts.append(0)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(EF_LABELS, mae_per_bin, color=C_BAR, edgecolor="white",
                  alpha=0.85, width=0.55)

    for bar, cnt, mae in zip(bars, counts, mae_per_bin):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.1,
                f"{mae:.2f}\n(n={cnt})",
                ha="center", va="bottom", fontsize=9)

    ax.axhline(5, color=C_LINE, lw=1.5, ls="--", label="±5 clinical threshold")
    ax.legend(fontsize=9)
    _style_ax(ax,
              f"{MODEL_LABEL} — MAE by Clinical EF Category ({split_name})",
              "Clinical EF Category", "Mean Absolute Error (EF %)")
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, f"error_by_ef_range_{split_name.lower()}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Plot 5: Training curves from epoch log ─────────────────────────────────
def plot_training_curves():
    if not os.path.exists(LOG_PATH):
        print(f"  [SKIP] No log found at {LOG_PATH}")
        return

    epochs, tr_loss, vl_loss, tr_mae, vl_mae = [], [], [], [], []
    with open(LOG_PATH) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            tr_loss.append(float(row["train_loss"]))
            vl_loss.append(float(row["val_loss"]))
            tr_mae.append(float(row["train_MAE"]))
            vl_mae.append(float(row["val_MAE"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    ax1.plot(epochs, tr_loss, label="Train Loss", color=C_SCATTER, lw=1.8)
    ax1.plot(epochs, vl_loss, label="Val Loss",   color=C_LINE,    lw=1.8, ls="--")
    best_epoch = epochs[int(np.argmin(vl_loss))]
    ax1.axvline(best_epoch, color="gray", lw=1, ls=":", label=f"Best epoch ({best_epoch})")
    ax1.legend(fontsize=9)
    _style_ax(ax1, f"{MODEL_LABEL} — Loss Curve", "Epoch", "Huber Loss")

    # MAE curve
    ax2.plot(epochs, tr_mae, label="Train MAE", color=C_SCATTER, lw=1.8)
    ax2.plot(epochs, vl_mae, label="Val MAE",   color=C_LINE,    lw=1.8, ls="--")
    best_mae_epoch = epochs[int(np.argmin(vl_mae))]
    ax2.axvline(best_mae_epoch, color="gray", lw=1, ls=":",
                label=f"Best MAE epoch ({best_mae_epoch})")
    ax2.axhline(5, color="green", lw=1, ls="--", alpha=0.6, label="±5 threshold")
    ax2.legend(fontsize=9)
    _style_ax(ax2, f"{MODEL_LABEL} — MAE Curve", "Epoch", "MAE (EF %)")

    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "training_curves.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ───────────────────────────────────────────────────────────────────
def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  {MODEL_LABEL} — Evaluation & Plots")
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {CKPT_PATH}")
    print(f"  Output dir : {PLOT_DIR}")
    print(f"{'='*55}\n")

    for split_id, split_name in [(1, "Val"), (2, "Test")]:
        print(f"── {split_name} split ──")
        true, pred = run_inference(split_id, device)

        mae  = np.mean(np.abs(true - pred))
        rmse = np.sqrt(np.mean((true - pred)**2))
        r2   = 1 - np.sum((true-pred)**2) / np.sum((true-np.mean(true))**2)
        acc5 = (np.abs(true-pred) <= 5).mean() * 100
        acc3 = (np.abs(true-pred) <= 3).mean() * 100

        print(f"  Samples : {len(true)}")
        print(f"  MAE     : {mae:.3f}%")
        print(f"  RMSE    : {rmse:.3f}%")
        print(f"  R²      : {r2:.4f}")
        print(f"  Acc@5   : {acc5:.2f}%")
        print(f"  Acc@3   : {acc3:.2f}%")
        print()

        plot_true_vs_pred(true, pred, split_name)
        plot_residuals(true, pred, split_name)
        plot_bland_altman(true, pred, split_name)
        plot_error_by_ef_range(true, pred, split_name)
        print()

    print("── Training curves ──")
    plot_training_curves()

    print(f"\n  All plots saved to: {PLOT_DIR}\n")


if __name__ == "__main__":
    run_evaluation()
