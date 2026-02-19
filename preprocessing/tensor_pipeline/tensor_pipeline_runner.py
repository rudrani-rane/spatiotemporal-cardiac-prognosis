import os
import json
import yaml

from .tensor_builder import build_tensor
from .tensor_normalizer import normalize_tensor
from .tensor_saver import save_tensor

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

FRAME_DIR = os.path.join(PROJECT_ROOT, "data", "processed_frames")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "tensors")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "tensor_config.yaml")


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def process_video(video_id, config):

    video_folder = os.path.join(FRAME_DIR, video_id)

    if not os.path.exists(video_folder):
        return

    ed_path = os.path.join(video_folder, "ED.png")
    es_path = os.path.join(video_folder, "ES.png")

    if not os.path.exists(ed_path) or not os.path.exists(es_path):
        return

    frame_paths = [ed_path, es_path]

    tensor = build_tensor(frame_paths, config)

    if tensor is None:
        return

    tensor = normalize_tensor(tensor, config)

    save_path = os.path.join(OUTPUT_DIR, f"{video_id}.pt")
    save_tensor(tensor, save_path)


def run_pipeline():

    print("Creating ED–ES tensors...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    config = load_config()

    for video_id in os.listdir(FRAME_DIR):
        process_video(video_id, config)

    print("Tensor Creation Complete.")


if __name__ == "__main__":
    run_pipeline()
