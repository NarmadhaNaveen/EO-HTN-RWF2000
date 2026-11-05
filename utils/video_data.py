import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class RWFVideoDataset(Dataset):
    def __init__(self, root, split="train", frames=8, size=96):
        self.root = os.path.join(root, split)
        self.frames = frames
        self.size = size
        self.samples = []

        for class_name, label in [("NonFight", 0), ("Fight", 1)]:
            folder = os.path.join(self.root, class_name)
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"Missing folder: {folder}")
            for name in sorted(os.listdir(folder)):
                if name.lower().endswith((".avi", ".mp4", ".mov")):
                    self.samples.append((os.path.join(folder, name), label))

        if not self.samples:
            raise RuntimeError(f"No videos found in {self.root}")

        print(f"{split}: {len(self.samples)} videos")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open video: {path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise RuntimeError(f"Video has no readable frames: {path}")

        positions = np.linspace(0, max(total - 1, 0), self.frames).astype(int)
        images = []

        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
            ok, frame = cap.read()

            if not ok:
                frame = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.size, self.size))

            images.append(frame)

        cap.release()

        array = np.stack(images).astype(np.float32) / 255.0
        array = (array - 0.5) / 0.5
        tensor = torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()

        return tensor, torch.tensor(label, dtype=torch.long)
