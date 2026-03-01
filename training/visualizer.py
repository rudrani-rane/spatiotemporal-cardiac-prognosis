"""
EF Prediction Visualizer
========================
Run AFTER both models have been trained.

Usage:
    python -m training.visualizer

Reads:
    training/models/motion_diff/motion_diff_epoch_log.csv
    training/models/motion_diff/motion_diff_model.pt
    training/models/hybrid/hybrid_epoch_log.csv
    training/models/hybrid/hybrid_model.pth

Outputs:
    training/plots/motion_diff/         – per-model charts
    training/plots/hybrid/              – per-model charts
    training/plots/comparison/         – cross-model comparison charts
"""

import os
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")                    # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader

# ── Path Config ───────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MOTION_LOG   = os.path.join(PROJECT_ROOT, "training", "models", "motion_diff", "motion_diff_epoch_log.csv")
MOTION_MODEL = os.path.join(PROJECT_ROOT, "training", "models", "motion_diff", "motion_diff_model.pt")

HYBRID_LOG   = os.path.join(PROJECT_ROOT, "training", "models", "hybrid", "hybrid_epoch_log.csv")
HYBRID_MODEL = os.path.join(PROJECT_ROOT, "training", "models", "hybrid", "hybrid_model.pth")

PLOT_ROOT        = os.path.join(PROJECT_ROOT, "training", "plots")
MOTION_PLOT_DIR  = os.path.join(PLOT_ROOT, "motion_diff")
HYBRID_PLOT_DIR  = os.path.join(PLOT_ROOT, "hybrid")
COMPARE_PLOT_DIR = os.path.join(PLOT_ROOT, "comparison")

# ── Style ─────────────────────────────────────────────────────────────────

PALETTE = {
    "motion": "#4C8BF5",   # blue
    "hybrid": "#E8624A",   # orange-red
    "train":  "#2ECC71",   # green
    "val":    "#E74C3C",   # red
    "ref":    "#95A5A6",   # grey
}

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.35,
    "font.size":          10,
    "axes.titlesize":     12,
    "axes.titleweight":   "bold",
})


# ════════════════════════════════════════════════════════════════════════════
# 1. Load CSV logs
# ════════════════════════════════════════════════════════════════════════════

def load_log(path):
    """Returns dict of lists keyed by CSV column name."""
    data = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                data.setdefault(k, []).append(float(v))
    return data


def _epochs(log):
    return np.array(log["epoch"], dtype=int)


# ════════════════════════════════════════════════════════════════════════════
# 2. Collect final predictions from test split
# ════════════════════════════════════════════════════════════════════════════

def _get_predictions_motion(device):
    from training.dataset_loader import EchoDataset
    from training.models.motion_diff.motion_difference_model import MotionDifferenceModel

    model = MotionDifferenceModel()
    model.load_state_dict(torch.load(MOTION_MODEL, map_location=device))
    model.to(device).eval()

    loader = DataLoader(EchoDataset(split=2), batch_size=32,
                        shuffle=False, num_workers=0)
    all_preds, all_true = [], []
    with torch.no_grad():
        for x, y in loader:
            preds = model(x.to(device)).squeeze(-1)
            all_preds.append(preds.cpu().numpy())
            all_true.append(y.numpy())

    return np.concatenate(all_preds), np.concatenate(all_true)


def _get_predictions_hybrid(device):
    from training.hybrid_dataset_loader import HybridEchoDataset
    from training.models.hybrid.hybrid_model import HybridEFModel

    model = HybridEFModel()
    model.load_state_dict(torch.load(HYBRID_MODEL, map_location=device))
    model.to(device).eval()

    loader = DataLoader(HybridEchoDataset(split=2), batch_size=32,
                        shuffle=False, num_workers=0)
    all_preds, all_true = [], []
    with torch.no_grad():
        for x, y in loader:
            preds = model(x.to(device)).squeeze(-1)
            all_preds.append(preds.cpu().numpy())
            all_true.append(y.numpy())

    return np.concatenate(all_preds), np.concatenate(all_true)


# ════════════════════════════════════════════════════════════════════════════
# 3. Per-model visualisation helpers
# ════════════════════════════════════════════════════════════════════════════

def _savefig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.relpath(path, PROJECT_ROOT)}")


# ─── 3-A. Training Dynamics ──────────────────────────────────────────────

def plot_training_dynamics(log, out_dir, label, color):
    ep = _epochs(log)

    panels = [
        ("Loss",   "train_loss",  "val_loss"),
        ("MAE",    "train_MAE",   "val_MAE"),
        ("RMSE",   "train_RMSE",  "val_RMSE"),
        ("R²",     "train_R2",    "val_R2"),
        ("Acc@5",  "train_Acc5",  "val_Acc5"),
        ("Acc@3",  "train_Acc3",  "val_Acc3"),
        ("Acc@2",  "train_Acc2",  "val_Acc2"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()
    fig.suptitle(f"{label} – Training Dynamics", fontsize=14, fontweight="bold", y=1.01)

    for i, (title, tr_key, vl_key) in enumerate(panels):
        ax = axes[i]
        ax.plot(ep, log[tr_key], color=PALETTE["train"], label="Train", lw=1.8)
        ax.plot(ep, log[vl_key], color=PALETTE["val"],   label="Val",   lw=1.8, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=9)

    axes[-1].set_visible(False)     # 8th cell – hide
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "01_training_dynamics.png"))


# ─── 3-B. Predicted vs Actual ─────────────────────────────────────────────

def plot_pred_vs_actual(preds, true, out_dir, label, color):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(true, preds, alpha=0.35, s=18, color=color, edgecolors="none")
    lo, hi = min(true.min(), preds.min()) - 2, max(true.max(), preds.max()) + 2
    ax.plot([lo, hi], [lo, hi], color=PALETTE["ref"], lw=1.2, linestyle="--", label="Perfect")
    ax.set_xlabel("True EF")
    ax.set_ylabel("Predicted EF")
    ax.set_title(f"{label} – Predicted vs Actual EF")
    ax.legend()
    _savefig(fig, os.path.join(out_dir, "02_pred_vs_actual.png"))


# ─── 3-C. Error Histogram ────────────────────────────────────────────────

def plot_error_histogram(preds, true, out_dir, label, color):
    errors = preds - true
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(errors, bins=40, color=color, alpha=0.8, edgecolor="white")
    axes[0].axvline(0, color=PALETTE["ref"], linestyle="--", lw=1.2)
    axes[0].set_xlabel("Residual Error (Predicted − True)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"{label} – Residual Distribution")

    abs_err = np.abs(errors)
    axes[1].hist(abs_err, bins=40, color=color, alpha=0.8, edgecolor="white")
    axes[1].set_xlabel("Absolute Error")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"{label} – Absolute Error Distribution")

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "03_error_histogram.png"))


# ─── 3-D. Clinical accuracy bar ──────────────────────────────────────────

def plot_clinical_accuracy(preds, true, out_dir, label, color):
    abs_err = np.abs(preds - true)
    tolerances = [2, 3, 5]
    accs = [100 * (abs_err <= t).mean() for t in tolerances]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar([f"±{t} EF" for t in tolerances], accs, color=color,
                  alpha=0.85, edgecolor="white", width=0.5)
    ax.axhline(90, color=PALETTE["ref"], linestyle="--", lw=1, label="90% target")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_ylabel("% Predictions Within Tolerance")
    ax.set_title(f"{label} – Clinical Accuracy")
    ax.legend()
    _savefig(fig, os.path.join(out_dir, "04_clinical_accuracy.png"))


# ─── 3-E. Tight Range Accuracy Curve ─────────────────────────────────────

def plot_tight_range_curve(preds, true, out_dir, label, color):
    abs_err = np.abs(preds - true)
    thresholds = np.arange(0.5, 15.5, 0.5)
    accs = [100 * (abs_err <= t).mean() for t in thresholds]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, accs, color=color, lw=2.2, marker="o", markersize=4)
    ax.axhline(90, color=PALETTE["ref"], linestyle="--", lw=1, label="90% line")
    ax.axvline(3,  color="#f39c12",      linestyle=":",  lw=1.2, label="±3 EF")
    ax.axvline(5,  color="#8e44ad",      linestyle=":",  lw=1.2, label="±5 EF")
    ax.set_xlabel("EF Error Tolerance")
    ax.set_ylabel("% Predictions Within Tolerance")
    ax.set_title(f"{label} – Tight Range Accuracy Curve")
    ax.legend()
    _savefig(fig, os.path.join(out_dir, "05_tight_range_curve.png"))


# ─── 3-F. Bias & Residual Behaviour ──────────────────────────────────────

def plot_residual_analysis(preds, true, out_dir, label, color):
    residuals = preds - true
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residual vs True EF
    axes[0].scatter(true, residuals, alpha=0.35, s=15, color=color, edgecolors="none")
    axes[0].axhline(0, color=PALETTE["ref"], linestyle="--", lw=1.2)
    axes[0].set_xlabel("True EF")
    axes[0].set_ylabel("Residual (Pred − True)")
    axes[0].set_title(f"{label} – Residual vs True EF")

    # Residual vs Predicted EF
    axes[1].scatter(preds, residuals, alpha=0.35, s=15, color=color, edgecolors="none")
    axes[1].axhline(0, color=PALETTE["ref"], linestyle="--", lw=1.2)
    axes[1].set_xlabel("Predicted EF")
    axes[1].set_ylabel("Residual (Pred − True)")
    axes[1].set_title(f"{label} – Residual vs Predicted EF")

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "06_residual_analysis.png"))


# ─── 3-G. EF Range Performance ───────────────────────────────────────────

def plot_ef_range_performance(preds, true, out_dir, label, color):
    abs_err = np.abs(preds - true)

    ranges  = [("Low EF (<40)", true <  40),
               ("Mid EF (40–55)", (true >= 40) & (true <= 55)),
               ("High EF (>55)", true >  55)]

    categories = [r[0] for r in ranges]
    mae_vals   = [abs_err[mask].mean() if mask.sum() > 0 else 0 for _, mask in ranges]
    counts     = [mask.sum() for _, mask in ranges]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(categories, mae_vals, color=color, alpha=0.85,
                edgecolor="white", width=0.5)
    axes[0].set_ylabel("Mean Absolute Error")
    axes[0].set_title(f"{label} – MAE by EF Range")
    for i, v in enumerate(mae_vals):
        axes[0].text(i, v + 0.1, f"{v:.2f}", ha="center", va="bottom")

    axes[1].bar(categories, counts, color=color, alpha=0.6, edgecolor="white")
    axes[1].set_ylabel("Sample Count")
    axes[1].set_title(f"{label} – Sample Count by EF Range")
    for i, v in enumerate(counts):
        axes[1].text(i, v + 0.5, str(v), ha="center", va="bottom")

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "07_ef_range_performance.png"))


# ─── 3-H. Hard Cases ─────────────────────────────────────────────────────

def plot_hard_cases(preds, true, out_dir, label, color):
    abs_err   = np.abs(preds - true)
    hard_mask = abs_err > 5

    if hard_mask.sum() == 0:
        print(f"  {label}: No hard cases (error > 5 EF). Skipping hard-case plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    hard_true = true[hard_mask]
    hard_err  = abs_err[hard_mask]

    axes[0].scatter(hard_true, hard_err, alpha=0.5, s=20, color="#e74c3c")
    axes[0].axhline(5, color=PALETTE["ref"], linestyle="--", lw=1.2)
    axes[0].set_xlabel("True EF")
    axes[0].set_ylabel("Absolute Error")
    axes[0].set_title(f"{label} – Hard Cases (error > 5 EF) by True EF\n"
                      f"({hard_mask.sum()} hard / {len(true)} total = "
                      f"{100*hard_mask.mean():.1f}%)")

    axes[1].hist(hard_true, bins=20, color="#e74c3c", alpha=0.75, edgecolor="white")
    axes[1].set_xlabel("True EF")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"{label} – True EF distribution of hard cases")

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "08_hard_cases.png"))


# ─── 3-I. Epoch-wise Prediction Variance ─────────────────────────────────

def plot_epoch_variance(log, out_dir, label, color):
    """
    Proxy for stability: val_MAE std over a rolling 5-epoch window.
    """
    ep      = _epochs(log)
    val_mae = np.array(log["val_MAE"])

    window  = 5
    rolling_std = [
        val_mae[max(0, i - window + 1): i + 1].std()
        for i in range(len(val_mae))
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ep, rolling_std, color=color, lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Rolling Std of Val MAE (window=5)")
    ax.set_title(f"{label} – Epoch-wise Prediction Stability")
    _savefig(fig, os.path.join(out_dir, "09_epoch_variance.png"))


# ─── 3-J. EF Distribution Overlay ────────────────────────────────────────

def plot_ef_distribution(preds, true, out_dir, label, color):
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(
        min(true.min(), preds.min()) - 2,
        max(true.max(), preds.max()) + 2,
        50
    )
    ax.hist(true,  bins=bins, alpha=0.55, color=PALETTE["ref"], label="True EF",  edgecolor="white")
    ax.hist(preds, bins=bins, alpha=0.55, color=color,           label="Predicted EF", edgecolor="white")
    ax.set_xlabel("EF Value")
    ax.set_ylabel("Count")
    ax.set_title(f"{label} – True vs Predicted EF Distribution")
    ax.legend()
    _savefig(fig, os.path.join(out_dir, "10_ef_distribution.png"))


# ════════════════════════════════════════════════════════════════════════════
# 4. Cross-Model Comparison Charts
# ════════════════════════════════════════════════════════════════════════════

def _best_val(log, key):
    """Return best (min/max depending on metric) validation value."""
    vals = np.array(log[key])
    if key in ("val_R2", "val_Acc5", "val_Acc3", "val_Acc2"):
        return float(vals.max())
    return float(vals.min())


def plot_metric_comparison(motion_log, hybrid_log, out_dir):
    metrics = [
        ("MAE",   "val_MAE",  False, "lower is better"),
        ("RMSE",  "val_RMSE", False, "lower is better"),
        ("R²",    "val_R2",   True,  "higher is better"),
        ("Acc@5", "val_Acc5", True,  "higher is better"),
        ("Acc@3", "val_Acc3", True,  "higher is better"),
        ("Acc@2", "val_Acc2", True,  "higher is better"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    fig.suptitle("Cross-Model Comparison – Best Validation Metrics",
                 fontsize=14, fontweight="bold")

    for i, (name, key, higher_better, note) in enumerate(metrics):
        ax = axes[i]
        motion_val = _best_val(motion_log, key)
        hybrid_val = _best_val(hybrid_log, key)

        vals   = [motion_val, hybrid_val]
        colors = [PALETTE["motion"], PALETTE["hybrid"]]
        labels = ["Motion\nDiff", "Hybrid\nEF"]

        bars = ax.bar(labels, vals, color=colors, alpha=0.85,
                      edgecolor="white", width=0.45)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005 * max(vals),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=10)
        ax.set_title(f"{name}\n({note})", fontsize=10)
        ax.set_ylim(0, max(vals) * 1.2 if max(vals) > 0 else 1)

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "01_metric_comparison.png"))


def plot_accuracy_comparison(motion_preds, motion_true,
                              hybrid_preds, hybrid_true, out_dir):
    """Grouped bar chart for clinical accuracy at ±2, ±3, ±5 EF."""
    tolerances = [2, 3, 5]
    motion_accs = [100 * (np.abs(motion_preds - motion_true) <= t).mean() for t in tolerances]
    hybrid_accs = [100 * (np.abs(hybrid_preds - hybrid_true) <= t).mean() for t in tolerances]

    x      = np.arange(len(tolerances))
    width  = 0.32
    labels = [f"±{t} EF" for t in tolerances]

    fig, ax = plt.subplots(figsize=(9, 6))
    b1 = ax.bar(x - width / 2, motion_accs, width, label="Motion Diff",
                color=PALETTE["motion"], alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + width / 2, hybrid_accs, width, label="Hybrid EF",
                color=PALETTE["hybrid"], alpha=0.85, edgecolor="white")

    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

    ax.axhline(90, color=PALETTE["ref"], linestyle="--", lw=1.2, label="90% target")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("% Predictions Within Tolerance")
    ax.set_title("Cross-Model – Clinical Accuracy Comparison")
    ax.set_ylim(0, 115)
    ax.legend()
    _savefig(fig, os.path.join(out_dir, "02_accuracy_comparison.png"))


def plot_distribution_overlay(motion_preds, motion_true,
                               hybrid_preds, hybrid_true, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    all_vals = np.concatenate([motion_true, hybrid_true, motion_preds, hybrid_preds])
    bins = np.linspace(all_vals.min() - 2, all_vals.max() + 2, 50)

    # True EF
    axes[0].hist(motion_true, bins, alpha=0.6, color=PALETTE["ref"], label="True EF (Motion)", edgecolor="w")
    axes[0].hist(hybrid_true,  bins, alpha=0.4, color="#7F8C8D",       label="True EF (Hybrid)", edgecolor="w")
    axes[0].set_title("True EF – Both Datasets")
    axes[0].legend(fontsize=7)
    axes[0].set_xlabel("EF")

    # Predicted EF overlay
    axes[1].hist(motion_preds, bins, alpha=0.55, color=PALETTE["motion"], label="Motion Diff Pred", edgecolor="w")
    axes[1].hist(hybrid_preds,  bins, alpha=0.55, color=PALETTE["hybrid"],  label="Hybrid EF Pred",  edgecolor="w")
    axes[1].set_title("Predicted EF – Both Models")
    axes[1].legend(fontsize=7)
    axes[1].set_xlabel("EF")

    # Absolute errors
    m_err = np.abs(motion_preds - motion_true)
    h_err = np.abs(hybrid_preds - hybrid_true)
    err_bins = np.linspace(0, max(m_err.max(), h_err.max()) + 1, 40)
    axes[2].hist(m_err, err_bins, alpha=0.55, color=PALETTE["motion"], label="Motion Diff", edgecolor="w")
    axes[2].hist(h_err, err_bins, alpha=0.55, color=PALETTE["hybrid"],  label="Hybrid EF",  edgecolor="w")
    axes[2].set_title("Absolute Error Distribution")
    axes[2].legend(fontsize=7)
    axes[2].set_xlabel("Absolute Error")

    fig.suptitle("Cross-Model Distribution Comparison", fontweight="bold")
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "03_distribution_overlay.png"))


def plot_tight_range_comparison(motion_preds, motion_true,
                                 hybrid_preds, hybrid_true, out_dir):
    thresholds = np.arange(0.5, 15.5, 0.5)
    m_accs = [100 * (np.abs(motion_preds - motion_true) <= t).mean() for t in thresholds]
    h_accs = [100 * (np.abs(hybrid_preds - hybrid_true) <= t).mean() for t in thresholds]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, m_accs, color=PALETTE["motion"], lw=2.2,
            label="Motion Diff", marker="o", markersize=4)
    ax.plot(thresholds, h_accs, color=PALETTE["hybrid"],  lw=2.2,
            label="Hybrid EF",   marker="s", markersize=4)
    ax.axhline(90, color=PALETTE["ref"], linestyle="--", lw=1, label="90% line")
    ax.axvline(3,  color="#f39c12", linestyle=":", lw=1.2, label="±3 EF")
    ax.axvline(5,  color="#8e44ad", linestyle=":", lw=1.2, label="±5 EF")
    ax.set_xlabel("EF Error Tolerance")
    ax.set_ylabel("% Predictions Within Tolerance")
    ax.set_title("Cross-Model – Tight Range Accuracy Comparison")
    ax.legend()
    _savefig(fig, os.path.join(out_dir, "04_tight_range_comparison.png"))


# ════════════════════════════════════════════════════════════════════════════
# 5. Run all visualizations
# ════════════════════════════════════════════════════════════════════════════

def run_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  EF Prediction Visualizer")
    print(f"  Device : {device}")
    print(f"{'='*55}\n")

    # ── Load logs ──
    print("Loading epoch logs…")
    if not os.path.exists(MOTION_LOG):
        print(f"  ✗ Motion Diff log not found: {MOTION_LOG}")
        motion_log = None
    else:
        motion_log = load_log(MOTION_LOG)
        print(f"  ✔ Motion Diff log  – {len(motion_log['epoch'])} epochs")

    if not os.path.exists(HYBRID_LOG):
        print(f"  ✗ Hybrid log not found: {HYBRID_LOG}")
        hybrid_log = None
    else:
        hybrid_log = load_log(HYBRID_LOG)
        print(f"  ✔ Hybrid log       – {len(hybrid_log['epoch'])} epochs")

    # ── Load predictions ──
    print("\nLoading model predictions on test split…")
    motion_preds = motion_true = hybrid_preds = hybrid_true = None

    if os.path.exists(MOTION_MODEL):
        try:
            motion_preds, motion_true = _get_predictions_motion(device)
            print(f"  ✔ Motion Diff test preds – {len(motion_true)} samples")
        except Exception as e:
            print(f"  ✗ Motion Diff preds failed: {e}")
    else:
        print(f"  ✗ Motion Diff model not found: {MOTION_MODEL}")

    if os.path.exists(HYBRID_MODEL):
        try:
            hybrid_preds, hybrid_true = _get_predictions_hybrid(device)
            print(f"  ✔ Hybrid test preds      – {len(hybrid_true)} samples")
        except Exception as e:
            print(f"  ✗ Hybrid preds failed: {e}")
    else:
        print(f"  ✗ Hybrid model not found: {HYBRID_MODEL}")

    # ══════════════════════════════
    # Motion Diff Plots
    # ══════════════════════════════
    if motion_log is not None:
        print(f"\nGenerating Motion Diff charts → {MOTION_PLOT_DIR}")
        plot_training_dynamics(motion_log, MOTION_PLOT_DIR, "Motion Diff", PALETTE["motion"])
        plot_epoch_variance(motion_log, MOTION_PLOT_DIR, "Motion Diff", PALETTE["motion"])

    if motion_preds is not None:
        plot_pred_vs_actual(motion_preds, motion_true, MOTION_PLOT_DIR, "Motion Diff", PALETTE["motion"])
        plot_error_histogram(motion_preds, motion_true, MOTION_PLOT_DIR, "Motion Diff", PALETTE["motion"])
        plot_clinical_accuracy(motion_preds, motion_true, MOTION_PLOT_DIR, "Motion Diff", PALETTE["motion"])
        plot_tight_range_curve(motion_preds, motion_true, MOTION_PLOT_DIR, "Motion Diff", PALETTE["motion"])
        plot_residual_analysis(motion_preds, motion_true, MOTION_PLOT_DIR, "Motion Diff", PALETTE["motion"])
        plot_ef_range_performance(motion_preds, motion_true, MOTION_PLOT_DIR, "Motion Diff", PALETTE["motion"])
        plot_hard_cases(motion_preds, motion_true, MOTION_PLOT_DIR, "Motion Diff", PALETTE["motion"])
        plot_ef_distribution(motion_preds, motion_true, MOTION_PLOT_DIR, "Motion Diff", PALETTE["motion"])

    # ══════════════════════════════
    # Hybrid Plots
    # ══════════════════════════════
    if hybrid_log is not None:
        print(f"\nGenerating Hybrid charts → {HYBRID_PLOT_DIR}")
        plot_training_dynamics(hybrid_log, HYBRID_PLOT_DIR, "Hybrid EF", PALETTE["hybrid"])
        plot_epoch_variance(hybrid_log, HYBRID_PLOT_DIR, "Hybrid EF", PALETTE["hybrid"])

    if hybrid_preds is not None:
        plot_pred_vs_actual(hybrid_preds, hybrid_true, HYBRID_PLOT_DIR, "Hybrid EF", PALETTE["hybrid"])
        plot_error_histogram(hybrid_preds, hybrid_true, HYBRID_PLOT_DIR, "Hybrid EF", PALETTE["hybrid"])
        plot_clinical_accuracy(hybrid_preds, hybrid_true, HYBRID_PLOT_DIR, "Hybrid EF", PALETTE["hybrid"])
        plot_tight_range_curve(hybrid_preds, hybrid_true, HYBRID_PLOT_DIR, "Hybrid EF", PALETTE["hybrid"])
        plot_residual_analysis(hybrid_preds, hybrid_true, HYBRID_PLOT_DIR, "Hybrid EF", PALETTE["hybrid"])
        plot_ef_range_performance(hybrid_preds, hybrid_true, HYBRID_PLOT_DIR, "Hybrid EF", PALETTE["hybrid"])
        plot_hard_cases(hybrid_preds, hybrid_true, HYBRID_PLOT_DIR, "Hybrid EF", PALETTE["hybrid"])
        plot_ef_distribution(hybrid_preds, hybrid_true, HYBRID_PLOT_DIR, "Hybrid EF", PALETTE["hybrid"])

    # ══════════════════════════════
    # Cross-Model Comparison
    # ══════════════════════════════
    if motion_log is not None and hybrid_log is not None:
        print(f"\nGenerating comparison charts → {COMPARE_PLOT_DIR}")
        plot_metric_comparison(motion_log, hybrid_log, COMPARE_PLOT_DIR)

    if motion_preds is not None and hybrid_preds is not None:
        plot_accuracy_comparison(motion_preds, motion_true,
                                  hybrid_preds, hybrid_true, COMPARE_PLOT_DIR)
        plot_distribution_overlay(motion_preds, motion_true,
                                   hybrid_preds, hybrid_true, COMPARE_PLOT_DIR)
        plot_tight_range_comparison(motion_preds, motion_true,
                                     hybrid_preds, hybrid_true, COMPARE_PLOT_DIR)

    print(f"\n{'='*55}")
    print(f"  All visualizations complete.")
    print(f"  Plots saved under: training/plots/")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    run_all()
