import os
import torch
import pandas as pd
from torch.utils.data import Dataset


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../")
)

TENSOR_DIR = os.path.join(PROJECT_ROOT, "data", "tensors")
LABEL_PATH = os.path.join(PROJECT_ROOT, "data", "metadata", "video_labels.csv")


class EchoDataset(Dataset):
    def __init__(self, split):
        """
        split:
        0 = train
        1 = val
        2 = test
        """

        self.labels = pd.read_csv(LABEL_PATH)

        # filter by split
        self.labels = self.labels[self.labels["split"] == split]

        self.video_ids = self.labels["video_id"].values
        self.targets = self.labels["EF"].values

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):

        video_id = self.video_ids[idx]
        ef = self.targets[idx]

        tensor_path = os.path.join(TENSOR_DIR, f"{video_id}.pt")

        tensor = torch.load(tensor_path)

        return tensor.float(), torch.tensor(ef).float()