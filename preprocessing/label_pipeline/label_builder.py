import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

FILELIST_PATH = os.path.join(PROJECT_ROOT, "EchoNet-Dynamic", "FileList.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "metadata", "video_labels.csv")


def build_labels():

    print("Loading FileList.csv...")

    df = pd.read_csv(FILELIST_PATH)

    # Remove .avi extension if present
    df["video_id"] = df["FileName"].str.replace(".avi", "", regex=False)

    print("Calculating Stroke Volume (SV)...")

    df["SV"] = df["EDV"] - df["ESV"]

    labels = df[[
        "video_id",
        "EF",
        "ESV",
        "EDV",
        "SV",
        "Split"
    ]].copy()

    labels.rename(columns={"Split": "split"}, inplace=True)

    # Encode split into numeric classes
    split_mapping = {
        "TRAIN": 0,
        "VAL": 1,
        "TEST": 2
    }

    labels["split"] = labels["split"].map(split_mapping)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    labels.to_csv(OUTPUT_PATH, index=False)

    print(f"Labels saved at: {OUTPUT_PATH}")
    print(f"Total labels created: {len(labels)}")


if __name__ == "__main__":
    build_labels()
