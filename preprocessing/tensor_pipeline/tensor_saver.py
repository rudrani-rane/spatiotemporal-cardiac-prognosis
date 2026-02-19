import torch
import os


def save_tensor(tensor, save_path):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save(torch.tensor(tensor), save_path)
