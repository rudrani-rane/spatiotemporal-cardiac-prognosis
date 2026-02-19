import os
import json
import cv2
import pandas as pd
import random

# Project root
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

VIDEOS_DIR = os.path.join(PROJECT_ROOT, "EchoNet-Dynamic", "Videos")
TRACING_CSV = os.path.join(PROJECT_ROOT, "EchoNet-Dynamic", "VolumeTracings.csv")
FILELIST_CSV = os.path.join(PROJECT_ROOT, "EchoNet-Dynamic", "FileList.csv")
FRAME_MAP_JSON = os.path.join(PROJECT_ROOT, "data", "mappings", "frame_index_map.json")

SAVE_DIR = os.path.join(PROJECT_ROOT, "data", "debug_frames")
os.makedirs(SAVE_DIR, exist_ok=True)


def extract_frame(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def overlay_tracing_lines(frame, tracing_rows):
    for _, row in tracing_rows.iterrows():
        x1, y1 = int(row["X1"]), int(row["Y1"])
        x2, y2 = int(row["X2"]), int(row["Y2"])
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return frame


def validate_mapping():

    print("Loading mapping...")
    with open(FRAME_MAP_JSON, 'r') as f:
        mapping = json.load(f)

    tracing_df = pd.read_csv(TRACING_CSV)
    filelist_df = pd.read_csv(FILELIST_CSV)

    # Check dataset alignment
    valid_videos = set(filelist_df["FileName"].str.replace(".avi", "", regex=False).values)

    sample_videos = random.sample(list(mapping.keys()), 3)

    for video in sample_videos:

        print(f"\nValidating: {video}")

        video_id = video.replace(".avi", "")
        if video_id not in valid_videos:

            print("⚠ Video not in FileList — skipping")
            continue

        video_path = os.path.join(VIDEOS_DIR, video)

        frames = mapping[video]

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        for frame_no in frames:

            if frame_no >= total_frames:
                print(f"⚠ Frame {frame_no} exceeds video length ({total_frames})")
                continue

            frame = extract_frame(video_path, frame_no)

            if frame is None:
                print(f"⚠ Failed to extract frame {frame_no}")
                continue

            tracing = tracing_df[
                (tracing_df["FileName"] == video) &
                (tracing_df["Frame"] == frame_no)
            ]

            if tracing.empty:
                print(f"⚠ No tracing found for frame {frame_no}")
                continue

            frame = overlay_tracing_lines(frame, tracing)

            save_path = os.path.join(
                SAVE_DIR,
                f"{video}_frame{frame_no}.png"
            )

            cv2.imwrite(save_path, frame)

            print(f"✔ Saved: {save_path}")


if __name__ == "__main__":
    validate_mapping()
