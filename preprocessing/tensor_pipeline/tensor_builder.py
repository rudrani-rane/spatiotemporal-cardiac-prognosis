import os
import cv2
import numpy as np


def load_and_resize_frame(frame_path, config):

    img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    img = cv2.resize(
        img,
        (config["resize_width"], config["resize_height"])
    )

    return img


def build_tensor(frame_paths, config):

    frames = []

    for path in frame_paths:
        img = load_and_resize_frame(path, config)
        if img is not None:
            frames.append(img)

    if len(frames) == 0:
        return None

    return np.stack(frames)
