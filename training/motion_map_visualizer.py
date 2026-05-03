"""
Motion Map Visualizer
=====================
Generates motion map visualizations (ED frame | ES frame | |ED-ES| heatmap)
for all trained models on test-split samples.

Usage:
    python -m training.motion_map_visualizer

Outputs:
    training/plots/motion_diff/11_motion_maps.png
    training/plots/hybrid/11_motion_maps.png
    training/plots/cardio_cnn/11_motion_maps.png
    training/plots/resnet_18/11_motion_maps.png
    training/plots/comparison/05_avg_motion_maps.png
"""

import os
import sys
import csv

# Ensure project root is on sys.path so 'training.*' imports work whether
# the script is run from the repo root (python -m training.motion_map_visualizer)
# or directly from inside the training/ folder (python motion_map_visualizer.py).
_PROJECT_ROOT_EARLY = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT_EARLY not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT_EARLY)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ── Path Config ───────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FRAMES_DIR   = os.path.join(PROJECT_ROOT, "data", "processed_frames")
TENSOR_DIR   = os.path.join(PROJECT_ROOT, "data", "tensors")
LABEL_PATH   = os.path.join(PROJECT_ROOT, "data", "metadata", "video_labels.csv")
PLOT_ROOT    = os.path.join(PROJECT_ROOT, "training", "plots")

IMG_SIZE  = 112
N_SAMPLES = 12   # samples to visualise per model grid

# ── Model Registry ────────────────────────────────────────────────────────

MODELS = {
    "motion_diff": {
        "label":      "Motion Diff",
        "color":      "#4C8BF5",
        "model_path": os.path.join(PROJECT_ROOT, "training", "models",
                                   "motion_diff", "motion_diff_model.pt"),
        "plot_dir":   os.path.join(PLOT_ROOT, "motion_diff"),
        "input_type": "tensor",   # loads [ED, ES] from data/tensors/
        "normalize":  False,
    },
    "hybrid": {
        "label":      "Hybrid EF",
        "color":      "#E8624A",
        "model_path": os.path.join(PROJECT_ROOT, "training", "models",
                                   "hybrid", "hybrid_model.pth"),
        "plot_dir":   os.path.join(PLOT_ROOT, "hybrid"),
        "input_type": "frames",
        "normalize":  False,      # HybridEchoDataset uses raw [0,1] values
    },
    "cardio_cnn": {
        "label":      "Cardio CNN",
        "color":      "#2ECC71",
        "model_path": os.path.join(PROJECT_ROOT, "training", "models",
                                   "cardio_cnn", "cnn_model.pth"),
        "plot_dir":   os.path.join(PLOT_ROOT, "cardio_cnn"),
        "input_type": "frames",
        "normalize":  True,       # CNNEchoDataset applies ImageNet normalisation
    },
    "resnet_18": {
        "label":      "ResNet-18",
        "color":      "#9B59B6",
        "model_path": os.path.join(PROJECT_ROOT, "training", "models",
                                   "resnet_18", "resnet18_model.pth"),
        "plot_dir":   os.path.join(PLOT_ROOT, "resnet_18"),
        "input_type": "frames",
        "normalize":  True,       # ResNetEchoDataset applies ImageNet normalisation
    },
}

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Style ─────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi":        130,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         8,
})


# ════════════════════════════════════════════════════════════════════════════
# 1. Data helpers
# ════════════════════════════════════════════════════════════════════════════

def load_test_samples():
    """Return list of (video_id, ef) for the test split (split == 2)."""
    samples = []
    with open(LABEL_PATH, newline="") as f:
        for row in csv.DictReader(f):
            row = {k.strip().lower(): v.strip() for k, v in row.items()}
            try:
                if int(row.get("split", -1)) != 2:
                    continue
            except ValueError:
                continue
            vid  = row.get("video_id", "").strip()
            ef_s = row.get("ef", "").strip()
            if not vid or not ef_s:
                continue
            try:
                samples.append((vid, float(ef_s)))
            except ValueError:
                pass
    return samples


def pick_diverse_samples(samples, n):
    """Select n samples spread across low / mid / high EF ranges."""
    rng   = np.random.default_rng(seed=42)
    low   = [(v, e) for v, e in samples if e <  40]
    mid   = [(v, e) for v, e in samples if 40 <= e <= 55]
    high  = [(v, e) for v, e in samples if e >  55]
    result = []
    per_group = max(1, n // 3)
    for group in (low, mid, high):
        if group:
            idx = rng.choice(len(group), size=min(per_group, len(group)), replace=False)
            result.extend(group[i] for i in idx)
    # fill to n if needed
    used = {v for v, _ in result}
    for v, e in samples:
        if len(result) >= n:
            break
        if v not in used:
            result.append((v, e))
            used.add(v)
    return result[:n]


def _load_gray_png(path):
    """Load a PNG as grayscale float32 [H, W] in [0, 1]."""
    img = Image.open(path).convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def load_ed_es(video_id, input_type):
    """
    Return (ed, es) as float32 numpy arrays [H, W] in [0, 1].
    Falls back to zero arrays if files are missing.
    """
    if input_type == "tensor":
        path = os.path.join(TENSOR_DIR, f"{video_id}.pt")
        if os.path.exists(path):
            t = torch.load(path, map_location="cpu").float()  # [2, H, W]
            return t[0].numpy(), t[1].numpy()
        return (np.zeros((IMG_SIZE, IMG_SIZE), np.float32),
                np.zeros((IMG_SIZE, IMG_SIZE), np.float32))

    # frame-based
    frame_dir = os.path.join(FRAMES_DIR, video_id)
    ed_path   = os.path.join(frame_dir, "ED.png")
    es_path   = os.path.join(frame_dir, "ES.png")
    if os.path.exists(ed_path) and os.path.exists(es_path):
        return _load_gray_png(ed_path), _load_gray_png(es_path)

    # fallback: first / last sorted PNG in the folder
    if os.path.isdir(frame_dir):
        pngs = sorted(f for f in os.listdir(frame_dir) if f.endswith(".png"))
        if len(pngs) >= 2:
            return (_load_gray_png(os.path.join(frame_dir, pngs[0])),
                    _load_gray_png(os.path.join(frame_dir, pngs[-1])))

    return (np.zeros((IMG_SIZE, IMG_SIZE), np.float32),
            np.zeros((IMG_SIZE, IMG_SIZE), np.float32))


# ════════════════════════════════════════════════════════════════════════════
# 2. Model loading & inference
# ════════════════════════════════════════════════════════════════════════════

def load_model(model_key, device):
    """Instantiate and load weights for the given model key."""
    info = MODELS[model_key]
    if not os.path.exists(info["model_path"]):
        return None

    if model_key == "motion_diff":
        from training.models.motion_diff.motion_difference_model import MotionDifferenceModel
        m = MotionDifferenceModel()
    elif model_key == "hybrid":
        from training.models.hybrid.hybrid_model import HybridEFModel
        m = HybridEFModel()
    elif model_key == "cardio_cnn":
        from training.models.cardio_cnn.cnn_model import CardioCNNModel
        m = CardioCNNModel()
    elif model_key == "resnet_18":
        from training.models.resnet_18.resnet18_model import ResNet18EFModel
        m = ResNet18EFModel(pretrained=True)
    else:
        return None

    m.load_state_dict(torch.load(info["model_path"], map_location=device))
    m.to(device).eval()
    return m


def _norm_channel(arr):
    """Apply ImageNet normalisation to a single [H, W] float32 array → tensor [H, W]."""
    from torchvision.transforms import Normalize
    t    = torch.from_numpy(arr).unsqueeze(0)    # [1, H, W]
    t3   = t.repeat(3, 1, 1)                     # [3, H, W]
    norm = Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
    return norm(t3)[0]                           # [H, W]


def predict_ef(model, model_key, ed, es, device):
    """Run a single inference for one (ED, ES) pair; return float EF prediction."""
    normalize = MODELS[model_key]["normalize"]
    motion    = np.abs(ed - es)
    with torch.no_grad():
        if model_key == "motion_diff":
            x = torch.from_numpy(
                np.stack([ed, es], axis=0)
            ).unsqueeze(0).to(device)                                # [1, 2, H, W]
        elif normalize:
            x = torch.stack(
                [_norm_channel(ed), _norm_channel(es), _norm_channel(motion)], dim=0
            ).unsqueeze(0).to(device)                                # [1, 3, H, W]
        else:
            x = torch.from_numpy(
                np.stack([ed, es, motion], axis=0)
            ).unsqueeze(0).to(device)                                # [1, 3, H, W]
        return model(x).squeeze().item()


# ════════════════════════════════════════════════════════════════════════════
# 3. Plotting helpers
# ════════════════════════════════════════════════════════════════════════════

def _savefig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.relpath(path, PROJECT_ROOT)}")


def plot_motion_maps_per_model(samples, model, model_key, model_info, device):
    """
    Grid: N rows × 3 cols  [ED frame | ES frame | Motion Map |ED-ES|]
    Each row corresponds to one test sample.
    Predictions are annotated in the row label if the model loaded successfully.
    """
    n          = len(samples)
    label      = model_info["label"]
    input_type = model_info["input_type"]

    fig, axes = plt.subplots(n, 3, figsize=(9, n * 2.4))
    if n == 1:
        axes = axes[np.newaxis, :]   # keep consistent 2-D indexing

    fig.suptitle(f"{label} – Motion Maps (Test Split, N={n})",
                 fontsize=13, fontweight="bold", y=1.005)

    for col, title in enumerate(["ED Frame", "ES Frame", "Motion Map |ED−ES|"]):
        axes[0, col].set_title(title, fontsize=9, fontweight="bold")

    last_im = None
    for row, (vid, true_ef) in enumerate(samples):
        ed, es = load_ed_es(vid, input_type)
        motion = np.abs(ed - es)

        pred_ef = None
        if model is not None:
            try:
                pred_ef = predict_ef(model, model_key, ed, es, device)
            except Exception:
                pass

        row_lbl = f"{vid[:14]}\nTrue EF={true_ef:.1f}"
        if pred_ef is not None:
            err = pred_ef - true_ef
            row_lbl += f"\nPred={pred_ef:.1f}  Err={err:+.1f}"

        axes[row, 0].imshow(ed,     cmap="gray", vmin=0, vmax=1)
        axes[row, 1].imshow(es,     cmap="gray", vmin=0, vmax=1)
        last_im = axes[row, 2].imshow(motion, cmap="hot", vmin=0, vmax=1)

        for ax in axes[row]:
            ax.axis("off")

        # Annotate left of the row using figure-space text (avoids tight_layout warning)
        axes[row, 0].text(
            -0.08, 0.5, row_lbl,
            fontsize=7, ha="right", va="center",
            transform=axes[row, 0].transAxes,
            multialignment="right",
        )

    if last_im is not None:
        plt.colorbar(last_im, ax=axes[:, 2], fraction=0.04, pad=0.02,
                     label="Motion Intensity")

    # Use subplots_adjust instead of tight_layout to avoid warnings from
    # out-of-axes row labels placed via transAxes text.
    fig.subplots_adjust(left=0.22, right=0.95, top=0.96, bottom=0.02,
                        hspace=0.08, wspace=0.05)
    _savefig(fig, os.path.join(model_info["plot_dir"], "11_motion_maps.png"))


def plot_avg_motion_comparison(all_samples):
    """
    Cross-model grid:
        Rows = models,  Cols = EF ranges (Low / Mid / High)
    Each cell shows the average motion map across up to 50 test samples in that EF range.
    The motion map is always computed from raw frames/tensors (independent of model).
    """
    ef_ranges = [
        ("Low EF\n(<40)",   lambda e: e <  40),
        ("Mid EF\n(40–55)", lambda e: 40 <= e <= 55),
        ("High EF\n(>55)",  lambda e: e >  55),
    ]

    model_keys = list(MODELS.keys())
    n_rows = len(model_keys)
    n_cols = len(ef_ranges)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.5, n_rows * 3.2))
    fig.suptitle("Average Motion Maps by EF Range – All Models",
                 fontsize=13, fontweight="bold")

    # Column headers
    for col_i, (range_label, _) in enumerate(ef_ranges):
        axes[0, col_i].set_title(range_label, fontsize=10, fontweight="bold")

    # Pre-bucket samples
    buckets = {
        col_i: [(v, e) for v, e in all_samples if fn(e)]
        for col_i, (_, fn) in enumerate(ef_ranges)
    }

    for row_i, model_key in enumerate(model_keys):
        input_type = MODELS[model_key]["input_type"]
        axes[row_i, 0].set_ylabel(MODELS[model_key]["label"],
                                  fontsize=9, rotation=90, labelpad=10)

        for col_i in range(n_cols):
            bucket = buckets[col_i][:50]
            maps   = []
            for vid, _ in bucket:
                ed, es = load_ed_es(vid, input_type)
                maps.append(np.abs(ed - es))

            avg_map = (np.mean(maps, axis=0)
                       if maps
                       else np.zeros((IMG_SIZE, IMG_SIZE), np.float32))
            vmax    = float(avg_map.max()) or 1.0

            axes[row_i, col_i].imshow(avg_map, cmap="hot", vmin=0, vmax=vmax)
            axes[row_i, col_i].axis("off")
            axes[row_i, col_i].text(
                0.5, -0.04, f"n={len(bucket)}",
                ha="center", va="top", fontsize=7,
                transform=axes[row_i, col_i].transAxes,
            )

    fig.tight_layout()
    _savefig(fig, os.path.join(PLOT_ROOT, "comparison", "05_avg_motion_maps.png"))


# ════════════════════════════════════════════════════════════════════════════
# 4. Main
# ════════════════════════════════════════════════════════════════════════════

def run_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  Motion Map Visualizer")
    print(f"  Device : {device}")
    print(f"{'='*55}\n")

    all_test = load_test_samples()
    print(f"Test samples: {len(all_test)}")

    diverse = pick_diverse_samples(all_test, N_SAMPLES)
    print(f"Selected {len(diverse)} diverse samples for per-model grids\n")

    for model_key, model_info in MODELS.items():
        label = model_info["label"]
        print(f"── {label} {'─' * 40}")

        if not os.path.exists(model_info["model_path"]):
            print(f"  ✗ Model weights not found: {model_info['model_path']}")
            continue

        model = None
        try:
            model = load_model(model_key, device)
            print(f"  ✔ Loaded model weights")
        except Exception as e:
            print(f"  ✗ Failed to load model: {e}")

        plot_motion_maps_per_model(diverse, model, model_key, model_info, device)

    print("\n── Cross-Model Average Motion Maps ──────────────────")
    plot_avg_motion_comparison(all_test)

    print("\nDone.")


if __name__ == "__main__":
    run_all()
