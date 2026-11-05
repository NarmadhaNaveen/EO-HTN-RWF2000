import argparse
import cv2
import numpy as np
import torch
from models.eo_htn import EOHTN

def load_video(path, frames, size):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: " + path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError("Video contains no readable frames.")

    positions = np.linspace(0, total - 1, frames).astype(int)
    images = []

    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((size, size, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (size, size))
        images.append(frame)

    cap.release()
    arr = np.stack(images).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    args = parser.parse_args()

    checkpoint = torch.load(
        "checkpoints/best_eo_htn.pth", map_location="cpu", weights_only=False
    )
    model = EOHTN(frames=checkpoint["frames"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    video = load_video(
        args.video, checkpoint["frames"], checkpoint["size"]
    )

    with torch.no_grad():
        probs = torch.softmax(model(video), dim=1)[0]

    label = int(probs.argmax())
    print("Prediction:", "Fight" if label == 1 else "NonFight")
    print("NonFight probability:", round(float(probs[0]), 4))
    print("Fight probability:", round(float(probs[1]), 4))

if __name__ == "__main__":
    main()

