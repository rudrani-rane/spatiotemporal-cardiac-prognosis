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


class CNNEchoDataset(Dataset):
    """
    Dataset for the pure-CNN cardiac EF model.

    Actual CSV format:
        video_id, EF, ESV, EDV, SV, split
        'split' is numeric: 0=train  1=val  2=test

    Frame layout:
        processed_frames/<video_id>/ed_frame.png
        processed_frames/<video_id>/es_frame.png

    Returns 3-channel tensor [3, IMG_SIZE, IMG_SIZE]:
        ch-0 : ED frame
        ch-1 : ES frame
        ch-2 : Motion map = |ED - ES|
    """

    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]

    def __init__(self, split: int = 0):
        """
        Args:
            split : 0 = train, 1 = val, 2 = test  (matches numeric CSV column)
        """
        assert split in (0, 1, 2), f"split must be 0/1/2, got {split}"
        self.split   = split
        self.samples = []          # [(video_id, ef_float), ...]
        self._load_csv()

    # ── CSV ────────────────────────────────────────────────────────────────
    def _load_csv(self):
        with open(LABEL_PATH, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Strip whitespace and lowercase all keys for robustness
                row = {k.strip().lower(): v.strip() for k, v in row.items()}

                # 'split' column holds an integer (0 / 1 / 2)
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

    
    def _load_gray_tensor(self, path: str) -> torch.Tensor:
        """Load image as grayscale float [1, H, W] in [0, 1]."""
        img = Image.open(path).convert("L")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)   # [1, H, W]


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_id, ef = self.samples[idx]
        frame_dir    = os.path.join(VIDEO_DIR, video_id)

        ed     = self._load_gray_tensor(os.path.join(frame_dir, "ED.png"))
        es     = self._load_gray_tensor(os.path.join(frame_dir, "ES.png"))
        motion = torch.abs(ed - es)

        # Normalise each channel with ImageNet stats, then take single channel
        norm        = T.Normalize(mean=self._MEAN, std=self._STD)
        ed_norm     = norm(ed.repeat(3, 1, 1))[0]       # [H, W]
        es_norm     = norm(es.repeat(3, 1, 1))[0]       # [H, W]
        motion_norm = norm(motion.repeat(3, 1, 1))[0]   # [H, W]

        x = torch.stack([ed_norm, es_norm, motion_norm], dim=0)  # [3, H, W]
        y = torch.tensor(ef, dtype=torch.float32)
        return x, y



if __name__ == "__main__":
    for split_id, name in [(0, "TRAIN"), (1, "VAL"), (2, "TEST")]:
        try:
            ds = CNNEchoDataset(split=split_id)
            x, y = ds[0]
            print(f"[{name}] samples={len(ds)}  x={x.shape}  EF={y.item():.2f}")
        except RuntimeError as e:
            print(f"[{name}] SKIP — {e}")