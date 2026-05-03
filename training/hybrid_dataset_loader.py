import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# ====== PATH CONFIG ======
VIDEO_DIR = r"D:\Setu\SDP_Project\spatiotemporal-cardiac-prognosis\data\processed_frames"
LABEL_PATH = r"D:\Setu\SDP_Project\spatiotemporal-cardiac-prognosis\data\metadata\video_labels.csv"

IMG_SIZE = 112


class HybridEchoDataset(Dataset):
    """
    Returns:
        3-channel tensor [ED, ES, MOTION]
        EF value
    """

    def __init__(self, split=0):

        self.labels = pd.read_csv(LABEL_PATH)
        self.labels = self.labels[self.labels["split"] == split]

        self.video_ids = self.labels["video_id"].values
        self.efs = self.labels["EF"].values

    def __len__(self):
        return len(self.video_ids)

    def load_frame(self, frame_path):
        img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        return img.astype(np.float32)

    def __getitem__(self, idx):

        vid = self.video_ids[idx]
        ef = self.efs[idx]

        video_folder = os.path.join(VIDEO_DIR, vid)

        frames = sorted(os.listdir(video_folder))

        if len(frames) < 2:
            ed = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
            es = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        else:
            ed = self.load_frame(os.path.join(video_folder, frames[0]))
            es = self.load_frame(os.path.join(video_folder, frames[-1]))

        motion = np.abs(ed - es)

        stacked = np.stack([ed, es, motion], axis=0)

        return torch.tensor(stacked), torch.tensor(ef, dtype=torch.float32)