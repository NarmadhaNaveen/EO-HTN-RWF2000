import cv2
import numpy as np
import torch


def analyze_segments(
    video_path,
    model,
    number_of_frames,
    size,
    segment_seconds=2.0,
    stride_seconds=1.0,
    max_segments=30
):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Cannot open video for segment analysis.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0 or fps <= 0:
        cap.release()
        raise RuntimeError("Invalid video frame count or FPS.")

    duration = total_frames / fps
    window_frames = max(1, int(segment_seconds * fps))
    stride_frames = max(1, int(stride_seconds * fps))

    if total_frames <= window_frames:
        starts = [0]
    else:
        starts = list(range(
            0, total_frames - window_frames + 1, stride_frames
        ))
        final_start = total_frames - window_frames

        if starts[-1] != final_start:
            starts.append(final_start)

    if len(starts) > max_segments:
        selected = np.linspace(
            0, len(starts) - 1, max_segments
        ).round().astype(int)
        starts = [starts[i] for i in selected]

    results = []

    for start in starts:
        end = min(start + window_frames, total_frames - 1)

        positions = np.linspace(
            start, end, number_of_frames
        ).astype(int)

        images = []

        for position in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(position))
            ok, frame = cap.read()

            if not ok:
                frame = np.zeros(
                    (size, size, 3), dtype=np.uint8
                )
                rgb = frame
            else:
                rgb = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )
                rgb = cv2.resize(rgb, (size, size))

            images.append(rgb)

        array = np.stack(images).astype(np.float32) / 255.0
        array = (array - 0.5) / 0.5

        tensor = torch.from_numpy(array).permute(
            0, 3, 1, 2
        ).contiguous().unsqueeze(0)

        with torch.inference_mode():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        fight_probability = float(probabilities[1].item())

        results.append({
            "start": start / fps,
            "end": end / fps,
            "midpoint": ((start + end) / 2) / fps,
            "fight_probability": fight_probability * 100,
            "frames": images,
            "positions": positions.tolist()
        })

    cap.release()
    return results
