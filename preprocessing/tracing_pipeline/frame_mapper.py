import sys
import os

# Add project root to Python path BEFORE imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)

import json
from collections import defaultdict
from utils.csv_utils import load_csv_safe
from utils.path_utils import (
    initialize_data_folders,
    get_frame_mapping_json_path
)


def build_frame_index_map(volume_tracing_csv_path):
    """
    Builds mapping:
    VideoID -> List of Frames that contain LV tracing
    """

    print("Loading VolumeTracings.csv...")

    df = load_csv_safe(volume_tracing_csv_path)

    if df is None:
        raise ValueError("Failed to load VolumeTracings.csv")

    required_columns = ["FileName", "Frame"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    print("Processing frame mappings...")

    frame_map = defaultdict(set)

    for _, row in df.iterrows():
        video_id = row["FileName"]
        frame = int(row["Frame"])

        frame_map[video_id].add(frame)

    # Convert to sorted list
    frame_map = {
        video_id: sorted(list(frames))
        for video_id, frames in frame_map.items()
    }

    print(f"Processed {len(frame_map)} videos.")

    return frame_map


def save_frame_index_map(frame_map):
    """
    Saves mapping into JSON
    """

    initialize_data_folders()

    save_path = get_frame_mapping_json_path()

    with open(save_path, "w") as f:
        json.dump(frame_map, f, indent=4)

    print(f"Frame mapping saved to {save_path}")


def main():

    # Project root = SDP
    PROJECT_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../")
    )

    csv_path = os.path.join(
        PROJECT_ROOT,
        "EchoNet-Dynamic",
        "VolumeTracings.csv"
    )

    frame_map = build_frame_index_map(csv_path)
    save_frame_index_map(frame_map)



if __name__ == "__main__":
    main()
