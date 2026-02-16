import os

# Root data directory
BASE_DATA_DIR = "data"

# Subdirectories
PROCESSED_DIR = os.path.join(BASE_DATA_DIR, "processed")
METADATA_DIR = os.path.join(BASE_DATA_DIR, "metadata")
MAPPINGS_DIR = os.path.join(BASE_DATA_DIR, "mappings")


def create_dir_if_not_exists(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)


def initialize_data_folders():
    """Ensure all required data folders exist"""
    create_dir_if_not_exists(BASE_DATA_DIR)
    create_dir_if_not_exists(PROCESSED_DIR)
    create_dir_if_not_exists(METADATA_DIR)
    create_dir_if_not_exists(MAPPINGS_DIR)


def get_processed_video_path(video_id):
    """Return save path for processed tensor"""
    return os.path.join(PROCESSED_DIR, f"{video_id}.pt")


def get_metadata_csv_path():
    """Return metadata.csv path"""
    return os.path.join(METADATA_DIR, "metadata.csv")


def get_frame_mapping_json_path():
    """Return frame index mapping path"""
    return os.path.join(MAPPINGS_DIR, "frame_index_map.json")
