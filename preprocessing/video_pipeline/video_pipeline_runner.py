import os
import yaml
import cv2
from tqdm import tqdm

from .video_loader import load_video, extract_frame, release_video
from .frame_sampler import load_frame_mapping, get_ed_es_frames
from .video_preprocessor import preprocess_frame


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

VIDEOS_DIR = os.path.join(PROJECT_ROOT, "EchoNet-Dynamic", "Videos")
MAPPING_PATH = os.path.join(PROJECT_ROOT, "data", "mappings", "frame_index_map.json")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed_frames")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "video_config.yaml")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)


def run_video_pipeline():

    print("Loading config...")
    config = load_config()

    print("Loading frame mapping...")
    mapping = load_frame_mapping(MAPPING_PATH)

    print("Processing videos...")

    for video_name in tqdm(mapping.keys()):

        video_path = os.path.join(VIDEOS_DIR, video_name)

        if not os.path.exists(video_path):
            print(f"⚠ Skipping missing video: {video_name}")
            continue

        try:
            ed_frame, es_frame = get_ed_es_frames(video_name, mapping)
        except Exception as e:
            print(f"⚠ Mapping issue for {video_name}: {e}")
            continue

        cap = load_video(video_path)

        frames_to_extract = {
            "ED": ed_frame,
            "ES": es_frame
        }

        for label, frame_no in frames_to_extract.items():

            frame = extract_frame(cap, frame_no)

            if frame is None:
                print(f"⚠ Failed to extract {label} frame for {video_name}")
                continue

            frame = preprocess_frame(frame, config)

            save_folder = os.path.join(OUTPUT_DIR, video_name.replace(".avi", ""))
            os.makedirs(save_folder, exist_ok=True)

            save_path = os.path.join(save_folder, f"{label}.png")

            cv2.imwrite(save_path, (frame * 255).astype("uint8"))

        release_video(cap)

    print("Video Pipeline Completed")


if __name__ == "__main__":
    run_video_pipeline()
