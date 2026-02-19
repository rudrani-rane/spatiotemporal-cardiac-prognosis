def sample_temporal_window(anchor_frame, total_frames, config):
    T = config["frames_per_clip"]
    stride = config["stride"]

    half = T // 2

    indices = []

    for i in range(-half, half + 1):
        frame_id = anchor_frame + (i * stride)

        frame_id = max(0, frame_id)
        frame_id = min(frame_id, total_frames - 1)

        indices.append(frame_id)

    return sorted(indices)
