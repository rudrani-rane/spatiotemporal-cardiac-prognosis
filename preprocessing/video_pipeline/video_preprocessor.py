import cv2


def preprocess_frame(frame, config):
    if config["grayscale"]:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame = cv2.resize(frame, (config["img_size"], config["img_size"]))

    if config["normalize"]:
        frame = frame / 255.0

    return frame
