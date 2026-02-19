import cv2
import os


def load_video(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")
    
    return cap


def get_total_frames(cap):
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def extract_frame(cap, frame_number):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    
    if not ret:
        return None
    
    return frame


def release_video(cap):
    cap.release()
