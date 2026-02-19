import json
import os


def load_frame_mapping(mapping_path):
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
    
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    
    return mapping


def get_ed_es_frames(video_name, mapping):
    if video_name not in mapping:
        raise KeyError(f"{video_name} not found in frame mapping")
    
    frames = mapping[video_name]
    
    if len(frames) != 2:
        raise ValueError(f"Invalid frame mapping for {video_name}")
    
    return frames[0], frames[1]
