"""
efnet_dataset_loader.py
───────────────────────
Dataset loader for EF-Net.

Key difference from CNN/ResNet loaders:
    Instead of loading 2 frames (ED + ES), this loader reads ALL frames
    stored in the video folder, sorts them in temporal order, and samples
    a fixed-length clip of NUM_FRAMES frames.

Expected frame layout on disk:
    processed_frames/
      <video_id>/
        frame_0000.png   ← first frame (or whatever naming convention)
        frame_0001.png
        ...
        ed_frame.png     ← EchoNet-style named frames also handled
        es_frame.png

    The loader auto-detects the naming convention by listing the folder.

Sampling strategies (set via SAMPLE_MODE):
    'uniform'  — picks NUM_FRAMES evenly spaced across the full clip
                 (best for inference / validation — deterministic)
    'random'   — picks a random contiguous window of NUM_FRAMES frames
                 (train-time augmentation — adds temporal variety)

Returns:
    x : torch.Tensor [T, H, W]   — grayscale clip (float32, normalised)
    y : torch.Tensor []           — LVEF value (float32 scalar)
"""

import os
import re
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
NUM_FRAMES  = 16      # frames per clip fed to EF-Net
IMG_SIZE    = 112     # spatial resolution (must match model)
SAMPLE_MODE = "uniform"   # 'uniform' or 'random'
# ══════════════════════════════════════════════════════════════════════════

# Grayscale mean/std (computed from EchoNet-Dynamic statistics)
# Using 0.5/0.5 is the standard for grayscale medical images:
# maps [0, 1] → [-1, 1] which works well with echocardiogram pixel intensities
_MEAN = [0.5]
_STD  = [0.5]


class EFNetEchoDataset(Dataset):
    """
    Video clip dataset for EF-Net.

    CSV format (same as other loaders):
        video_id, EF, ESV, EDV, SV, split
        split: 0=train, 1=val, 2=test

    Clip construction:
        1. List all .png / .jpg files in the video folder
        2. Sort them by embedded number (natural sort)
        3. Sample NUM_FRAMES frames using uniform or random strategy
        4. Load, resize, normalise each frame
        5. Stack into [T, H, W] tensor
    """

    def __init__(self, split: int = 0, sample_mode: str = SAMPLE_MODE):
        """
        Args:
            split       : 0=train, 1=val, 2=test
            sample_mode : 'uniform' (deterministic) or 'random' (augmentation)
        """
        assert split in (0, 1, 2), f"split must be 0/1/2, got {split}"
        assert sample_mode in ("uniform", "random"), \
            f"sample_mode must be 'uniform' or 'random', got {sample_mode}"

        self.split       = split
        self.sample_mode = sample_mode
        # Use random sampling only for training; always uniform for val/test
        self.use_random  = (sample_mode == "random") and (split == 0)
        self.samples     = []   # [(video_id, ef), ...]
        self._load_csv()

    # ── CSV ────────────────────────────────────────────────────────────────
    def _load_csv(self):
        with open(LABEL_PATH, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
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
                f"  Check that 'video_id' folders exist inside VIDEO_DIR."
            )

    # ── Frame discovery ────────────────────────────────────────────────────
    @staticmethod
    def _natural_key(s: str) -> list:
        """
        Natural sort key: splits string into text and numeric parts so that
        'frame_0010.png' sorts after 'frame_0009.png' (not 'frame_0009.png'
        after 'frame_00100.png').
        """
        return [int(c) if c.isdigit() else c.lower()
                for c in re.split(r'(\d+)', s)]

    def _get_frame_paths(self, frame_dir: str) -> list:
        """
        Returns sorted list of all image file paths in frame_dir.
        Handles mixed naming: frame_0000.png, 0000.png, ed_frame.png, etc.
        Excludes non-image files automatically.
        """
        valid_ext = {".png", ".jpg", ".jpeg"}
        files = [
            f for f in os.listdir(frame_dir)
            if os.path.splitext(f)[1].lower() in valid_ext
        ]
        files.sort(key=self._natural_key)
        return [os.path.join(frame_dir, f) for f in files]

    # ── Frame sampling ─────────────────────────────────────────────────────
    def _sample_indices(self, total_frames: int) -> list:
        """
        Sample NUM_FRAMES indices from a clip of total_frames frames.

        If total_frames < NUM_FRAMES: indices are repeated (loop) to fill.
        If total_frames >= NUM_FRAMES:
            uniform → evenly spaced indices
            random  → random contiguous window
        """
        if total_frames == 0:
            return [0] * NUM_FRAMES

        if total_frames < NUM_FRAMES:
            # Repeat frames cyclically to reach NUM_FRAMES
            indices = [i % total_frames for i in range(NUM_FRAMES)]
            return indices

        if self.use_random:
            # Random contiguous window
            max_start = total_frames - NUM_FRAMES
            start     = torch.randint(0, max_start + 1, (1,)).item()
            return list(range(start, start + NUM_FRAMES))
        else:
            # Uniform: evenly spaced, always deterministic
            step    = total_frames / NUM_FRAMES
            indices = [int(i * step) for i in range(NUM_FRAMES)]
            return indices

    # ── Single frame loading ───────────────────────────────────────────────
    def _load_frame(self, path: str) -> np.ndarray:
        """Load one frame as grayscale float32 [H, W] in [0, 1]."""
        img = Image.open(path).convert("L")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0

    # ── Dataset interface ──────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_id, ef = self.samples[idx]
        frame_dir    = os.path.join(VIDEO_DIR, video_id)

        # 1. Discover and sort all frames in this video folder
        all_paths = self._get_frame_paths(frame_dir)

        # 2. Select NUM_FRAMES indices
        indices = self._sample_indices(len(all_paths))

        # 3. Load selected frames
        frames = []
        for i in indices:
            frames.append(self._load_frame(all_paths[i]))

        # 4. Stack → [T, H, W] numpy array
        clip = np.stack(frames, axis=0).astype(np.float32)  # [T, H, W]

        # 5. Convert to tensor
        x = torch.from_numpy(clip)   # [T, H, W]

        # 6. Normalise: [0,1] → [-1,1]  (grayscale: each frame independently)
        #    T.Normalize expects [C, H, W] so we treat T as C for this op,
        #    then the model unsqueezes the channel dim internally.
        x = T.Normalize(mean=_MEAN * NUM_FRAMES,
                         std=_STD  * NUM_FRAMES)(x)  # [T, H, W]

        y = torch.tensor(ef, dtype=torch.float32)
        return x, y


# ── Quick sanity-check ─────────────────────────────────────────────────────
if __name__ == "__main__":
    for split_id, name in [(0, "TRAIN"), (1, "VAL"), (2, "TEST")]:
        try:
            ds  = EFNetEchoDataset(split=split_id, sample_mode="uniform")
            x, y = ds[0]
            print(f"[{name}] samples={len(ds)}  "
                  f"x={x.shape}  "
                  f"min={x.min():.2f}  max={x.max():.2f}  "
                  f"EF={y.item():.2f}")
        except RuntimeError as e:
            print(f"[{name}] SKIP — {e}")
