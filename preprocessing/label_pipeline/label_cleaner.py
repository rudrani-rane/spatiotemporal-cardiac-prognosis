import os
import json
import pandas as pd

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

LABELS_PATH = os.path.join(PROJECT_ROOT, "data", "metadata", "video_labels.csv")
MAPPING_PATH = os.path.join(PROJECT_ROOT, "data", "mappings", "frame_index_map.json")


def clean_labels():

    print("Loading labels...")
    labels = pd.read_csv(LABELS_PATH)

    print("Loading mapping...")
    with open(MAPPING_PATH, "r") as f:
        mapping = json.load(f)

    mapped_videos = set([k.replace(".avi", "") for k in mapping.keys()])

    original_count = len(labels)

    labels = labels[labels["video_id"].isin(mapped_videos)]

    new_count = len(labels)

    labels.to_csv(LABELS_PATH, index=False)

    print(f"Removed {original_count - new_count} invalid videos")
    print(f"Remaining valid videos: {new_count}")


if __name__ == "__main__":
    clean_labels()
