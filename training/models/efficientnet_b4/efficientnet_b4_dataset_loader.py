"""
efficientnet_b4_dataset_loader.py
──────────────────────────────────
Dataset loader for EfficientNet-B4.

Identical data pipeline to resnet18_dataset_loader.py:
    - ED frame  (ch-0)
    - ES frame  (ch-1)
    - Motion map |ED-ES|  (ch-2)
    - ImageNet normalisation
    - Optional augmentation (train only)

The only functional difference from the ResNet-18 loader is that
EfficientNet-B4 was trained at 380×380 resolution. We still use 112×112
because:
    1. All other models use 112 — keeps comparisons fair
    2. EfficientNet's AdaptiveAvgPool handles any input size
    3. Going to 380×380 would require 11× more GPU memory per batch

CSV format:
    video_id, EF, ESV, EDV, SV, split   (split: 0=train, 1=val, 2=test)

Frame layout:
    processed_frames/<video_id>/ed_frame.png
    processed_frames/<video_id>/es_frame.png
"""

import os
import csv
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


# ── Paths ──────────────────────────────────────────────────────────────────
VIDEO_DIR  = r"D:\Setu\SDP_Project\spatiotemporal-cardiac-prognosis\data\processed_frames"
LABEL_PATH = r"D:\Setu\SDP_Project\spatiotemporal-cardiac-prognosis\data\metadata\video_labels.csv"

# ══════════════════════════════════════════════════════════════════════════
#  HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════
IMG_SIZE = 112    # spatial resolution (model handles any size via AdaptiveAvgPool)
# ══════════════════════════════════════════════════════════════════════════

# Standard ImageNet normalisation — matches EfficientNet-B4 pretrained weights
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


class EfficientNetEchoDataset(Dataset):
    """
    Dataset for EfficientNet-B4 cardiac LVEF prediction.

    Returns 3-channel tensor [3, IMG_SIZE, IMG_SIZE]:
        ch-0 : ED frame   (End-Diastole,  ImageNet-normalised)
        ch-1 : ES frame   (End-Systole,   ImageNet-normalised)
        ch-2 : Motion map (|ED - ES|,     ImageNet-normalised)
    """

    def __init__(self, split: int = 0, augment: bool = False):
        """
        Args:
            split   : 0 = train, 1 = val, 2 = test
            augment : random flip + rotation (applied only when split=0)
        """
        assert split in (0, 1, 2), f"split must be 0/1/2, got {split}"

        self.split   = split
        self.augment = augment and (split == 0)   # never augment val/test
        self.samples = []
        self._load_csv()

    # ── CSV ────────────────────────────────────────────────────────────────
    def _load_csv(self):
        with open(LABEL_PATH, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Normalise all keys: lowercase + strip whitespace
                row = {k.strip().lower(): v.strip() for k, v in row.items()}

                try:
                    row_split = int(row.get("split", -1))
                except ValueError:
                    continue

                if row_split != self.split:
                    continue

                video_id = row.get("video_id", "").strip()
                ef_str   = row.get("ef", "").strip()

                if not video_id or not ef_str:
                    continue

                frame_dir = os.path.join(VIDEO_DIR, video_id)
                if not os.path.isdir(frame_dir):
                    continue

                try:
                    ef = float(ef_str)
                except ValueError:
                    continue

                self.samples.append((video_id, ef))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found for split={self.split}.\n"
                f"  VIDEO_DIR  = {VIDEO_DIR}\n"
                f"  LABEL_PATH = {LABEL_PATH}\n"
                f"  Check that 'video_id' folders exist in VIDEO_DIR and "
                f"that the CSV 'split' column contains integer {self.split}."
            )

    # ── Frame loading ──────────────────────────────────────────────────────
    def _load_gray(self, path: str) -> np.ndarray:
        """Load image as grayscale float32 numpy array [H, W] in [0, 1]."""
        img = Image.open(path).convert("L")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0

    # ── Dataset interface ──────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_id, ef = self.samples[idx]
        frame_dir    = os.path.join(VIDEO_DIR, video_id)

        # Load ED and ES frames as float32 [H, W] arrays in [0, 1]
        ed     = self._load_gray(os.path.join(frame_dir, "ED.png"))
        es     = self._load_gray(os.path.join(frame_dir, "ES.png"))

        # Motion channel: pixel-wise absolute difference
        # Large values where the heart wall moved between ED and ES
        motion = np.abs(ed - es)

        # Stack into [3, H, W] tensor
        x = torch.from_numpy(
            np.stack([ed, es, motion], axis=0).astype(np.float32)
        )

        # Optional augmentation — train split only
        if self.augment:
            # Random horizontal flip (50% chance)
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[2])

            # Random rotation ±10 degrees
            angle = (torch.rand(1).item() - 0.5) * 20.0
            x = T.functional.rotate(x, angle)

        # ImageNet normalisation (mean/std per channel)
        x = T.Normalize(mean=_MEAN, std=_STD)(x)   # [3, H, W]

        y = torch.tensor(ef, dtype=torch.float32)
        return x, y


# ── Quick sanity-check ─────────────────────────────────────────────────────
if __name__ == "__main__":
    for split_id, name in [(0, "TRAIN"), (1, "VAL"), (2, "TEST")]:
        try:
            ds   = EfficientNetEchoDataset(split=split_id,
                                           augment=(split_id == 0))
            x, y = ds[0]
            print(f"[{name}] samples={len(ds)}  "
                  f"x={x.shape}  "
                  f"min={x.min():.3f}  max={x.max():.3f}  "
                  f"EF={y.item():.2f}")
        except RuntimeError as e:
            print(f"[{name}] SKIP — {e}")
