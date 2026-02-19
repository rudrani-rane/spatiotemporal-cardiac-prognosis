import os
import json

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

MAPPING_PATH = os.path.join(PROJECT_ROOT, "data", "mappings", "frame_index_map.json")


def remove_missing_video():

    bad_video = "0X4F8859C8AB4DA9CB.avi"

    with open(MAPPING_PATH, "r") as f:
        mapping = json.load(f)

    if bad_video in mapping:
        del mapping[bad_video]
        print("Removed missing video from mapping")

    with open(MAPPING_PATH, "w") as f:
        json.dump(mapping, f, indent=4)


if __name__ == "__main__":
    remove_missing_video()
