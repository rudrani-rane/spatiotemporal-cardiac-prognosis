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

# CSV 'split' column is numeric: 0 = train, 1 = val, 2 = test

IMG_SIZE = 112


class ResNetEchoDataset(Dataset):
    """
    Dataset for the ResNet-18 cardiac EF model.

    Actual CSV format:
        video_id, EF, ESV, EDV, SV, split
        'split' is numeric: 0=train  1=val  2=test

    Frame layout:
        processed_frames/<video_id>/ed_frame.png
        processed_frames/<video_id>/es_frame.png

    Returns 3-channel tensor [3, IMG_SIZE, IMG_SIZE]:
        ch-0 : ED frame   (ImageNet-normalised)
        ch-1 : ES frame   (ImageNet-normalised)
        ch-2 : Motion map (ImageNet-normalised)
    """

    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]

    def __init__(self, split: int = 0, augment: bool = False):
        """
        Args:
            split   : 0 = train, 1 = val, 2 = test
            augment : random flip + rotation (train split only)
        """
        assert split in (0, 1, 2), f"split must be 0/1/2, got {split}"
        self.split   = split
        self.augment = augment and (split == 0)
        self.samples = []
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
                f"  Ensure 'video_id' folders exist in VIDEO_DIR and "
                f"that the CSV 'split' column contains integer {self.split}."
            )

    # ── Frame helpers ──────────────────────────────────────────────────────
    def _load_gray(self, path: str) -> np.ndarray:
        """Load image as grayscale float32 numpy [H, W] in [0, 1]."""
        img = Image.open(path).convert("L")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0

    # ── Dataset interface ──────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_id, ef = self.samples[idx]
        frame_dir    = os.path.join(VIDEO_DIR, video_id)

        ed     = self._load_gray(os.path.join(frame_dir, "ED.png"))
        es     = self._load_gray(os.path.join(frame_dir, "ES.png"))
        motion = np.abs(ed - es)

        # Stack → [3, H, W]
        x = torch.from_numpy(np.stack([ed, es, motion], axis=0))

        # Optional augmentation (train only)
        if self.augment:
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[2])
            angle = (torch.rand(1).item() - 0.5) * 20.0
            x = T.functional.rotate(x, angle)

        # ImageNet normalisation
        x = T.Normalize(mean=self._MEAN, std=self._STD)(x)

        y = torch.tensor(ef, dtype=torch.float32)
        return x, y


# ── Quick sanity-check ─────────────────────────────────────────────────────
if __name__ == "__main__":
    for split_id, name in [(0, "TRAIN"), (1, "VAL"), (2, "TEST")]:
        try:
            ds = ResNetEchoDataset(split=split_id, augment=(split_id == 0))
            x, y = ds[0]
            print(f"[{name}] samples={len(ds)}  x={x.shape}  EF={y.item():.2f}")
        except RuntimeError as e:
            print(f"[{name}] SKIP — {e}")
