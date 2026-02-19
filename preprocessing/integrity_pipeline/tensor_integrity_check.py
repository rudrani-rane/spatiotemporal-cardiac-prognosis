import os
import torch
import json
import pandas as pd

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

TENSOR_DIR = os.path.join(PROJECT_ROOT, "data", "tensors")
LABEL_PATH = os.path.join(PROJECT_ROOT, "data", "metadata", "video_labels.csv")
MAPPING_PATH = os.path.join(PROJECT_ROOT, "data", "mappings", "frame_index_map.json")


def run_check():

    print("Loading tensors...")
    tensor_files = set([
        f.replace(".pt", "")
        for f in os.listdir(TENSOR_DIR)
        if f.endswith(".pt")
    ])

    print("Loading labels...")
    labels = pd.read_csv(LABEL_PATH)
    label_videos = set(labels["video_id"].values)

    print("Loading mapping...")
    with open(MAPPING_PATH, "r") as f:
        mapping = json.load(f)

    mapped_videos = set([
        v.replace(".avi", "")
        for v in mapping.keys()
    ])

    # Alignment Checks
    print("\nChecking Tensor ↔ Labels alignment...")

    missing_tensor_for_labels = label_videos - tensor_files
    missing_labels_for_tensor = tensor_files - label_videos

    if missing_tensor_for_labels:
        print("⚠ Missing tensors for labelled videos:")
        print(missing_tensor_for_labels)

    if missing_labels_for_tensor:
        print("⚠ Extra tensors without labels:")
        print(missing_labels_for_tensor)

    if not missing_tensor_for_labels and not missing_labels_for_tensor:
        print("✔ Tensor ↔ Label alignment OK")

    print("\nChecking Tensor ↔ Mapping alignment...")

    missing_tensor_for_mapping = mapped_videos - tensor_files

    if missing_tensor_for_mapping:
        print("⚠ Missing tensors for mapped videos:")
        print(missing_tensor_for_mapping)
    else:
        print("✔ All mapped videos have tensors")

    # Tensor Quality Check
    print("\nChecking tensor structure...")

    bad_shape = []
    bad_range = []
    unreadable = []

    for vid in tensor_files:

        path = os.path.join(TENSOR_DIR, f"{vid}.pt")

        try:
            t = torch.load(path)

            if t.shape != (2, 112, 112):
                bad_shape.append(vid)

            if t.min() < 0 or t.max() > 1:
                bad_range.append(vid)

        except:
            unreadable.append(vid)

    if unreadable:
        print("⚠ Unreadable tensors:", unreadable)

    if bad_shape:
        print("⚠ Bad tensor shape:", bad_shape)

    if bad_range:
        print("⚠ Bad tensor value range:", bad_range)

    if not unreadable and not bad_shape and not bad_range:
        print("✔ Tensor quality OK")

    # Split Check
    print("\nChecking split balance...")

    filtered_labels = labels[labels["video_id"].isin(tensor_files)]

    split_counts = filtered_labels["split"].value_counts().sort_index()

    print(split_counts)

    print("\nTensor Integrity Check Completed.")


if __name__ == "__main__":
    run_check()
