"""
Temporal Echo Dataset
=====================
Loads the full cardiac cycle from raw AVI files in EchoNet-Dynamic/Videos/
and uniformly samples a fixed-length frame sequence for temporal modelling.

Returns:
    frames : FloatTensor [T, 1, 112, 112]  – grayscale, normalised [0, 1]
    ef     : FloatTensor []                – ejection fraction label
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# ── Path Config ───────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../")
)

VIDEO_DIR  = os.path.join(PROJECT_ROOT, "EchoNet-Dynamic", "Videos")
LABEL_PATH = os.path.join(PROJECT_ROOT, "data", "metadata", "video_labels.csv")

IMG_SIZE   = 112
SEQ_LEN    = 16   # frames uniformly sampled per video


class TemporalEchoDataset(Dataset):
    """
    Temporal dataset: samples SEQ_LEN frames uniformly from each AVI.

    Args:
        split    : 0 = train, 1 = val, 2 = test
        seq_len  : number of frames to sample (default 16)
        img_size : resize target (default 112)
    """

    def __init__(self, split: int, seq_len: int = SEQ_LEN, img_size: int = IMG_SIZE):
        self.seq_len  = seq_len
        self.img_size = img_size

        df = pd.read_csv(LABEL_PATH)
        df = df[df["split"] == split].reset_index(drop=True)

        # Only keep rows whose AVI file actually exists
        mask = df["video_id"].apply(
            lambda vid: os.path.isfile(os.path.join(VIDEO_DIR, vid + ".avi"))
        )
        df = df[mask].reset_index(drop=True)

        self.video_ids = df["video_id"].values
        self.efs       = df["EF"].values

    def __len__(self):
        return len(self.video_ids)

    def _load_sequence(self, vid: str) -> np.ndarray:
        """
        Opens AVI, uniformly samples self.seq_len frames,
        converts to grayscale and normalises to [0, 1].

        Returns:
            np.ndarray  [seq_len, img_size, img_size]  float32
        """
        path = os.path.join(VIDEO_DIR, vid + ".avi")
        cap  = cv2.VideoCapture(path)

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total = max(total, 1)

        # Uniform indices
        indices = np.linspace(0, total - 1, self.seq_len, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                frames.append(np.zeros((self.img_size, self.img_size), dtype=np.float32))
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (self.img_size, self.img_size))
            frames.append(gray.astype(np.float32) / 255.0)

        cap.release()

        # Pad if we couldn't read enough frames
        while len(frames) < self.seq_len:
            frames.append(np.zeros((self.img_size, self.img_size), dtype=np.float32))

        return np.stack(frames[:self.seq_len], axis=0)   # [T, H, W]

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        ef  = self.efs[idx]

        seq = self._load_sequence(vid)          # [T, H, W]  float32
        seq = seq[:, np.newaxis, :, :]          # [T, 1, H, W]

        return (
            torch.tensor(seq,  dtype=torch.float32),
            torch.tensor(ef,   dtype=torch.float32),
        )


if __name__ == "__main__":
    ds = TemporalEchoDataset(split=0)
    print(f"Train samples : {len(ds)}")
    x, y = ds[0]
    print(f"Tensor shape  : {x.shape}")   # [16, 1, 112, 112]
    print(f"EF label      : {y.item():.3f}")
