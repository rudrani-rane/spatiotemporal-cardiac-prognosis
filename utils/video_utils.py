import cv2


def is_video_readable(video_path):
    """Check if video can be opened"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    cap.release()
    return True


def get_frame_count(video_path):
    """Get total number of frames"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def get_video_fps(video_path):
    """Get video FPS"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps
