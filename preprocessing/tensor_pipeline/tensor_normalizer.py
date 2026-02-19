import numpy as np


def normalize_tensor(tensor, config):

    if config["normalization"] == "minmax":
        tensor = tensor.astype("float32") / 255.0

    return tensor
