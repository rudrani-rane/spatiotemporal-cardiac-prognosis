import os
import json
import pandas as pd

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

MAPPING_PATH = os.path.join(PROJECT_ROOT, "data", "mappings", "frame_index_map.json")
LABELS_PATH = os.path.join(PROJECT_ROOT, "data", "metadata", "video_labels.csv")
FRAMES_DIR = os.path.join(PROJECT_ROOT, "data", "processed_frames")


def load_data():
    print("Loading mapping...")
    with open(MAPPING_PATH, "r") as f:
        mapping = json.load(f)

    print("Loading labels...")
    labels = pd.read_csv(LABELS_PATH)

    return mapping, labels


def check_mapping_vs_labels(mapping, labels):
    print("\nChecking Mapping ↔ Labels alignment...")

    mapped_videos = set([k.replace(".avi", "") for k in mapping.keys()])
    labelled_videos = set(labels["video_id"])

    missing_in_labels = mapped_videos - labelled_videos
    missing_in_mapping = labelled_videos - mapped_videos

    print(f"Mapped videos: {len(mapped_videos)}")
    print(f"Labelled videos: {len(labelled_videos)}")

    print("\nVideos missing labels:")
    print(missing_in_labels)

    print("\nVideos missing mappings:")
    print(missing_in_mapping)

    if not missing_in_labels:
        print("✔ All mapped videos have labels")
    else:
        print(f"⚠ {len(missing_in_labels)} videos missing labels")

    if not missing_in_mapping:
        print("✔ All labelled videos have mappings")
    else:
        print(f"⚠ {len(missing_in_mapping)} videos missing mappings")


def check_frames_vs_mapping(mapping):
    print("\nChecking Frames ↔ Mapping alignment...")

    frame_dirs = set(os.listdir(FRAMES_DIR))
    mapped_videos = set([k.replace(".avi", "") for k in mapping.keys()])

    missing_frames = mapped_videos - frame_dirs
    extra_frames = frame_dirs - mapped_videos

    print("\nVideos missing frames:")
    print(missing_frames)

    if not missing_frames:
        print("✔ All mapped videos have extracted frames")
    else:
        print(f"⚠ {len(missing_frames)} videos missing frames")

    if not extra_frames:
        print("✔ No extra frame folders found")
    else:
        print(f"⚠ {len(extra_frames)} extra frame folders detected")


def check_split_balance(labels):
    print("\nChecking dataset split balance...")

    split_counts = labels["split"].value_counts().sort_index()

    print("Split distribution:")
    print(split_counts)

    total = len(labels)
    print("\nSplit %:")
    print((split_counts / total) * 100)


def run_integrity_check():
    mapping, labels = load_data()

    check_mapping_vs_labels(mapping, labels)
    check_frames_vs_mapping(mapping)
    check_split_balance(labels)

    print("\nPhase 1 Integrity Check Completed")


if __name__ == "__main__":
    run_integrity_check()
