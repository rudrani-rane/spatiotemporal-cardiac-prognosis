"""
Ensemble Visualizer
===================
Standalone visualizer for the Ensemble model and 3-way comparison.

Run AFTER both base models are trained and ensemble_runner has been executed.

Usage:
    python -m training.models.ensemble.ensemble_visualizer

Reads:
    training/models/motion_diff/motion_diff_model.pt
    training/models/motion_diff/motion_diff_epoch_log.csv
    training/models/hybrid/hybrid_model.pth
    training/models/hybrid/hybrid_epoch_log.csv

Outputs:
    training/plots/ensemble/          – per-model charts for the ensemble
    training/plots/comparison/        – updated 3-model comparison charts
"""

import os
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# ── Path Config ───────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..")
)

MOTION_LOG   = os.path.join(PROJECT_ROOT, "training", "models", "motion_diff", "motion_diff_epoch_log.csv")
MOTION_MODEL = os.path.join(PROJECT_ROOT, "training", "models", "motion_diff", "motion_diff_model.pt")

HYBRID_LOG   = os.path.join(PROJECT_ROOT, "training", "models", "hybrid", "hybrid_epoch_log.csv")
HYBRID_MODEL = os.path.join(PROJECT_ROOT, "training", "models", "hybrid", "hybrid_model.pth")

PLOT_ROOT         = os.path.join(PROJECT_ROOT, "training", "plots")
ENSEMBLE_PLOT_DIR = os.path.join(PLOT_ROOT, "ensemble")

# ── Style ─────────────────────────────────────────────────────────────────

PALETTE = {
    "motion":   "#4C8BF5",   # blue
    "hybrid":   "#E8624A",   # orange-red
    "ensemble": "#27AE60",   # green
    "ref":      "#95A5A6",   # grey
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
# 1. Data Loading
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


def _savefig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.relpath(path, PROJECT_ROOT)}")


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


def _get_predictions_ensemble(device):
    from training.models.ensemble.ensemble_predictor import EnsemblePredictor

    predictor = EnsemblePredictor(device=device)
    ens_preds, true_labels, _, _ = predictor.predict(split=2)
    return ens_preds, true_labels


# ════════════════════════════════════════════════════════════════════════════
# 2. Per-model plots  (ensemble only – reuse same plot patterns)
# ════════════════════════════════════════════════════════════════════════════

def plot_pred_vs_actual(preds, true, out_dir, label, color):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(true, preds, alpha=0.35, s=18, color=color, edgecolors="none")
    lo = min(true.min(), preds.min()) - 2
    hi = max(true.max(), preds.max()) + 2
    ax.plot([lo, hi], [lo, hi], color=PALETTE["ref"], lw=1.2,
            linestyle="--", label="Perfect")
    ax.set_xlabel("True EF")
    ax.set_ylabel("Predicted EF")
    ax.set_title(f"{label} – Predicted vs Actual EF")
    ax.legend()
    _savefig(fig, os.path.join(out_dir, "02_pred_vs_actual.png"))


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


def plot_clinical_accuracy(preds, true, out_dir, label, color):
    abs_err    = np.abs(preds - true)
    tolerances = [2, 3, 5]
    accs       = [100 * (abs_err <= t).mean() for t in tolerances]

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


def plot_tight_range_curve(preds, true, out_dir, label, color):
    abs_err    = np.abs(preds - true)
    thresholds = np.arange(0.5, 15.5, 0.5)
    accs       = [100 * (abs_err <= t).mean() for t in thresholds]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, accs, color=color, lw=2.2, marker="o", markersize=4)
    ax.axhline(90, color=PALETTE["ref"], linestyle="--", lw=1,   label="90% line")
    ax.axvline(3,  color="#f39c12",      linestyle=":",  lw=1.2, label="±3 EF")
    ax.axvline(5,  color="#8e44ad",      linestyle=":",  lw=1.2, label="±5 EF")
    ax.set_xlabel("EF Error Tolerance")
    ax.set_ylabel("% Predictions Within Tolerance")
    ax.set_title(f"{label} – Tight Range Accuracy Curve")
    ax.legend()
    _savefig(fig, os.path.join(out_dir, "05_tight_range_curve.png"))


def plot_residual_analysis(preds, true, out_dir, label, color):
    residuals = preds - true
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(true, residuals, alpha=0.35, s=15, color=color, edgecolors="none")
    axes[0].axhline(0, color=PALETTE["ref"], linestyle="--", lw=1.2)
    axes[0].set_xlabel("True EF")
    axes[0].set_ylabel("Residual (Pred − True)")
    axes[0].set_title(f"{label} – Residual vs True EF")

    axes[1].scatter(preds, residuals, alpha=0.35, s=15, color=color, edgecolors="none")
    axes[1].axhline(0, color=PALETTE["ref"], linestyle="--", lw=1.2)
    axes[1].set_xlabel("Predicted EF")
    axes[1].set_ylabel("Residual (Pred − True)")
    axes[1].set_title(f"{label} – Residual vs Predicted EF")

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "06_residual_analysis.png"))


def plot_ef_range_performance(preds, true, out_dir, label, color):
    abs_err = np.abs(preds - true)
    ranges  = [
        ("Low EF (<40)",    true < 40),
        ("Mid EF (40–55)",  (true >= 40) & (true <= 55)),
        ("High EF (>55)",   true > 55),
    ]
    categories = [r[0] for r in ranges]
    mae_vals   = [abs_err[mask].mean() if mask.sum() > 0 else 0 for _, mask in ranges]
    counts     = [mask.sum() for _, mask in ranges]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(categories, mae_vals, color=color, alpha=0.85, edgecolor="white", width=0.5)
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
    axes[0].set_title(
        f"{label} – Hard Cases (error > 5 EF) by True EF\n"
        f"({hard_mask.sum()} hard / {len(true)} total = {100*hard_mask.mean():.1f}%)"
    )

    axes[1].hist(hard_true, bins=20, color="#e74c3c", alpha=0.75, edgecolor="white")
    axes[1].set_xlabel("True EF")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"{label} – True EF distribution of hard cases")

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "08_hard_cases.png"))


def plot_ef_distribution(preds, true, out_dir, label, color):
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(
        min(true.min(), preds.min()) - 2,
        max(true.max(), preds.max()) + 2,
        50,
    )
    ax.hist(true,  bins=bins, alpha=0.55, color=PALETTE["ref"], label="True EF",      edgecolor="white")
    ax.hist(preds, bins=bins, alpha=0.55, color=color,          label="Predicted EF", edgecolor="white")
    ax.set_xlabel("EF Value")
    ax.set_ylabel("Count")
    ax.set_title(f"{label} – True vs Predicted EF Distribution")
    ax.legend()
    _savefig(fig, os.path.join(out_dir, "10_ef_distribution.png"))


# ════════════════════════════════════════════════════════════════════════════
# 3. 3-Model Comparison Charts  (overwrites training/plots/comparison/)
# ════════════════════════════════════════════════════════════════════════════

def plot_metric_comparison_3(motion_preds, motion_true,
                              hybrid_preds, hybrid_true,
                              ensemble_preds, ensemble_true,
                              out_dir):
    """3-model grouped bar chart – test-set metrics."""
    from training.metrics import compute_all_metrics

    def _metrics(p, t):
        return compute_all_metrics(torch.tensor(p), torch.tensor(t))

    m_met = _metrics(motion_preds,   motion_true)
    h_met = _metrics(hybrid_preds,   hybrid_true)
    e_met = _metrics(ensemble_preds, ensemble_true)

    metric_specs = [
        ("MAE",   "MAE",   False, "lower is better"),
        ("RMSE",  "RMSE",  False, "lower is better"),
        ("R²",    "R2",    True,  "higher is better"),
        ("Acc@5", "Acc@5", True,  "higher is better"),
        ("Acc@3", "Acc@3", True,  "higher is better"),
        ("Acc@2", "Acc@2", True,  "higher is better"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    fig.suptitle("3-Model Comparison – Test Set Metrics",
                 fontsize=14, fontweight="bold")

    colors = [PALETTE["motion"], PALETTE["hybrid"], PALETTE["ensemble"]]
    labels = ["Motion\nDiff", "Hybrid\nEF", "Ensemble"]

    for i, (name, key, higher_better, note) in enumerate(metric_specs):
        ax   = axes[i]
        vals = [m_met[key], h_met[key], e_met[key]]
        bars = ax.bar(labels, vals, color=colors, alpha=0.85,
                      edgecolor="white", width=0.45)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005 * max(vals),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(f"{name}\n({note})", fontsize=10)
        ax.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 1)

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "01_metric_comparison.png"))


def plot_accuracy_comparison_3(motion_preds, motion_true,
                                hybrid_preds, hybrid_true,
                                ensemble_preds, ensemble_true,
                                out_dir):
    """3-model grouped bar chart – clinical accuracy at ±2, ±3, ±5 EF."""
    tolerances    = [2, 3, 5]
    motion_accs   = [100 * (np.abs(motion_preds   - motion_true)   <= t).mean() for t in tolerances]
    hybrid_accs   = [100 * (np.abs(hybrid_preds   - hybrid_true)   <= t).mean() for t in tolerances]
    ensemble_accs = [100 * (np.abs(ensemble_preds - ensemble_true) <= t).mean() for t in tolerances]

    x      = np.arange(len(tolerances))
    width  = 0.24
    labels = [f"±{t} EF" for t in tolerances]

    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - width, motion_accs,   width, label="Motion Diff",
                color=PALETTE["motion"],   alpha=0.85, edgecolor="white")
    b2 = ax.bar(x,          hybrid_accs,   width, label="Hybrid EF",
                color=PALETTE["hybrid"],   alpha=0.85, edgecolor="white")
    b3 = ax.bar(x + width, ensemble_accs, width, label="Ensemble",
                color=PALETTE["ensemble"], alpha=0.85, edgecolor="white")

    for bars in (b1, b2, b3):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8)

    ax.axhline(90, color=PALETTE["ref"], linestyle="--", lw=1.2, label="90% target")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("% Predictions Within Tolerance")
    ax.set_title("3-Model – Clinical Accuracy Comparison (Test Set)")
    ax.set_ylim(0, 118)
    ax.legend()
    _savefig(fig, os.path.join(out_dir, "02_accuracy_comparison.png"))


def plot_distribution_overlay_3(motion_preds, motion_true,
                                 hybrid_preds, hybrid_true,
                                 ensemble_preds, ensemble_true,
                                 out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    all_vals = np.concatenate([motion_true, hybrid_true,
                                motion_preds, hybrid_preds, ensemble_preds])
    bins = np.linspace(all_vals.min() - 2, all_vals.max() + 2, 50)

    # True EF
    axes[0].hist(motion_true, bins, alpha=0.55, color=PALETTE["ref"],
                 label="True EF", edgecolor="w")
    axes[0].set_title("True EF Distribution (Test Set)")
    axes[0].legend(fontsize=7)
    axes[0].set_xlabel("EF")

    # Predicted EF – all 3 models
    axes[1].hist(motion_preds,   bins, alpha=0.5, color=PALETTE["motion"],
                 label="Motion Diff", edgecolor="w")
    axes[1].hist(hybrid_preds,   bins, alpha=0.5, color=PALETTE["hybrid"],
                 label="Hybrid EF",   edgecolor="w")
    axes[1].hist(ensemble_preds, bins, alpha=0.5, color=PALETTE["ensemble"],
                 label="Ensemble",    edgecolor="w")
    axes[1].set_title("Predicted EF – All 3 Models")
    axes[1].legend(fontsize=7)
    axes[1].set_xlabel("EF")

    # Absolute errors
    m_err    = np.abs(motion_preds   - motion_true)
    h_err    = np.abs(hybrid_preds   - hybrid_true)
    e_err    = np.abs(ensemble_preds - ensemble_true)
    err_max  = max(m_err.max(), h_err.max(), e_err.max())
    err_bins = np.linspace(0, err_max + 1, 40)
    axes[2].hist(m_err, err_bins, alpha=0.5, color=PALETTE["motion"],
                 label="Motion Diff", edgecolor="w")
    axes[2].hist(h_err, err_bins, alpha=0.5, color=PALETTE["hybrid"],
                 label="Hybrid EF",   edgecolor="w")
    axes[2].hist(e_err, err_bins, alpha=0.5, color=PALETTE["ensemble"],
                 label="Ensemble",    edgecolor="w")
    axes[2].set_title("Absolute Error Distribution")
    axes[2].legend(fontsize=7)
    axes[2].set_xlabel("Absolute Error")

    fig.suptitle("3-Model Distribution Comparison (Test Set)", fontweight="bold")
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "03_distribution_overlay.png"))


def plot_tight_range_comparison_3(motion_preds, motion_true,
                                   hybrid_preds, hybrid_true,
                                   ensemble_preds, ensemble_true,
                                   out_dir):
    thresholds = np.arange(0.5, 15.5, 0.5)
    m_accs = [100 * (np.abs(motion_preds   - motion_true)   <= t).mean() for t in thresholds]
    h_accs = [100 * (np.abs(hybrid_preds   - hybrid_true)   <= t).mean() for t in thresholds]
    e_accs = [100 * (np.abs(ensemble_preds - ensemble_true) <= t).mean() for t in thresholds]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(thresholds, m_accs, color=PALETTE["motion"],   lw=2.2,
            label="Motion Diff", marker="o", markersize=4)
    ax.plot(thresholds, h_accs, color=PALETTE["hybrid"],   lw=2.2,
            label="Hybrid EF",   marker="s", markersize=4)
    ax.plot(thresholds, e_accs, color=PALETTE["ensemble"], lw=2.2,
            label="Ensemble",    marker="^", markersize=4)
    ax.axhline(90, color=PALETTE["ref"], linestyle="--", lw=1,   label="90% line")
    ax.axvline(3,  color="#f39c12",      linestyle=":",  lw=1.2, label="±3 EF")
    ax.axvline(5,  color="#8e44ad",      linestyle=":",  lw=1.2, label="±5 EF")
    ax.set_xlabel("EF Error Tolerance")
    ax.set_ylabel("% Predictions Within Tolerance")
    ax.set_title("3-Model – Tight Range Accuracy Comparison (Test Set)")
    ax.legend()
    _savefig(fig, os.path.join(out_dir, "04_tight_range_comparison.png"))


# ════════════════════════════════════════════════════════════════════════════
# 4. Main
# ════════════════════════════════════════════════════════════════════════════

def run_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  Ensemble Visualizer")
    print(f"  Device : {device}")
    print(f"{'='*55}\n")

    # ── Load predictions ──────────────────────────────────────────────────
    print("Loading model predictions on test split…")

    motion_preds = motion_true = None
    hybrid_preds = hybrid_true = None
    ensemble_preds = ensemble_true = None

    try:
        motion_preds, motion_true = _get_predictions_motion(device)
        print(f"  ✔ Motion Diff  – {len(motion_true)} samples")
    except Exception as e:
        print(f"  ✗ Motion Diff failed: {e}")

    try:
        hybrid_preds, hybrid_true = _get_predictions_hybrid(device)
        print(f"  ✔ Hybrid EF    – {len(hybrid_true)} samples")
    except Exception as e:
        print(f"  ✗ Hybrid EF failed: {e}")

    try:
        ensemble_preds, ensemble_true = _get_predictions_ensemble(device)
        print(f"  ✔ Ensemble     – {len(ensemble_true)} samples")
    except Exception as e:
        print(f"  ✗ Ensemble failed: {e}")

    # ── Ensemble per-model plots ──────────────────────────────────────────
    if ensemble_preds is not None:
        print(f"\nGenerating Ensemble per-model charts → training/plots/ensemble/")
        plot_pred_vs_actual(ensemble_preds, ensemble_true,
                            ENSEMBLE_PLOT_DIR, "Ensemble", PALETTE["ensemble"])
        plot_error_histogram(ensemble_preds, ensemble_true,
                             ENSEMBLE_PLOT_DIR, "Ensemble", PALETTE["ensemble"])
        plot_clinical_accuracy(ensemble_preds, ensemble_true,
                               ENSEMBLE_PLOT_DIR, "Ensemble", PALETTE["ensemble"])
        plot_tight_range_curve(ensemble_preds, ensemble_true,
                               ENSEMBLE_PLOT_DIR, "Ensemble", PALETTE["ensemble"])
        plot_residual_analysis(ensemble_preds, ensemble_true,
                               ENSEMBLE_PLOT_DIR, "Ensemble", PALETTE["ensemble"])
        plot_ef_range_performance(ensemble_preds, ensemble_true,
                                  ENSEMBLE_PLOT_DIR, "Ensemble", PALETTE["ensemble"])
        plot_hard_cases(ensemble_preds, ensemble_true,
                        ENSEMBLE_PLOT_DIR, "Ensemble", PALETTE["ensemble"])
        plot_ef_distribution(ensemble_preds, ensemble_true,
                             ENSEMBLE_PLOT_DIR, "Ensemble", PALETTE["ensemble"])

    # ── 3-model comparison charts ─────────────────────────────────────────
    all_ready = (
        motion_preds is not None and
        hybrid_preds is not None and
        ensemble_preds is not None
    )

    if all_ready:
        print(f"\nGenerating 3-model comparison charts → training/plots/ensemble/")
        plot_metric_comparison_3(
            motion_preds, motion_true,
            hybrid_preds, hybrid_true,
            ensemble_preds, ensemble_true,
            ENSEMBLE_PLOT_DIR,
        )
        plot_accuracy_comparison_3(
            motion_preds, motion_true,
            hybrid_preds, hybrid_true,
            ensemble_preds, ensemble_true,
            ENSEMBLE_PLOT_DIR,
        )
        plot_distribution_overlay_3(
            motion_preds, motion_true,
            hybrid_preds, hybrid_true,
            ensemble_preds, ensemble_true,
            ENSEMBLE_PLOT_DIR,
        )
        plot_tight_range_comparison_3(
            motion_preds, motion_true,
            hybrid_preds, hybrid_true,
            ensemble_preds, ensemble_true,
            ENSEMBLE_PLOT_DIR,
        )
    else:
        print("\n  ⚠ Could not generate comparison charts – one or more models failed to load.")

    print(f"\n{'='*55}")
    print(f"  Ensemble visualization complete.")
    print(f"  All plots saved : training/plots/ensemble/")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    run_all()
