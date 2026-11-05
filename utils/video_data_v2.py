import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class RWFVideoDataset(Dataset):
    def __init__(self, root, split="train", frames=8, size=96):
        self.root = os.path.join(root, split)
        self.split = split
        self.frames = frames
        self.size = size
        self.training = split.lower() == "train"
        self.samples = []

        for class_name, label in [("NonFight", 0), ("Fight", 1)]:
            folder = os.path.join(self.root, class_name)

            if not os.path.isdir(folder):
                raise FileNotFoundError(f"Missing folder: {folder}")

            for name in sorted(os.listdir(folder)):
                if name.lower().endswith((".avi", ".mp4", ".mov")):
                    self.samples.append(
                        (os.path.join(folder, name), label)
                    )

        if not self.samples:
            raise RuntimeError(f"No videos found in {self.root}")

        print(f"{split}: {len(self.samples)} videos")

    def __len__(self):
        return len(self.samples)

    def _sample_positions(self, total):
        # Divide the video into temporal bins.
        # Training: randomly sample one frame per bin.
        # Validation: sample the centre of each bin.
        edges = np.linspace(0, total, self.frames + 1)
        positions = []

        for i in range(self.frames):
            start = int(np.floor(edges[i]))
            end = int(np.floor(edges[i + 1])) - 1

            start = min(start, total - 1)
            end = max(start, min(end, total - 1))

            if self.training:
                position = np.random.randint(start, end + 1)
            else:
                position = (start + end) // 2

            positions.append(position)

        return positions

    def _augment_clip(self, images):
        # Apply the same transformation to every frame
        # so temporal consistency is preserved.
        if not self.training:
            return images

        if np.random.random() < 0.5:
            images = [cv2.flip(frame, 1) for frame in images]

        if np.random.random() < 0.5:
            alpha = np.random.uniform(0.90, 1.10)
            beta = np.random.uniform(-10, 10)

            augmented = []
            for frame in images:
                frame = frame.astype(np.float32) * alpha + beta
                frame = np.clip(frame, 0, 255).astype(np.uint8)
                augmented.append(frame)

            images = augmented

        return images

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

        positions = self._sample_positions(total)
        images = []
        last_valid = None

        for position in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(position))
            ok, frame = cap.read()

            if not ok:
                if last_valid is None:
                    frame = np.zeros(
                        (self.size, self.size, 3), dtype=np.uint8
                    )
                else:
                    frame = last_valid.copy()
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(
                    frame, (self.size, self.size),
                    interpolation=cv2.INTER_AREA
                )
                last_valid = frame.copy()

            images.append(frame)

        cap.release()

        images = self._augment_clip(images)

        array = np.stack(images).astype(np.float32) / 255.0
        array = (array - 0.5) / 0.5

        tensor = torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()

        return tensor, torch.tensor(label, dtype=torch.long)
