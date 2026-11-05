import numpy as np
import torch
from torch.utils.data import DataLoader
from models.eo_htn import EOHTN
from utils.video_data import RWFVideoDataset

torch.set_num_threads(2)

root = "RWF-2000"
checkpoint = torch.load(
    "checkpoints/best_eo_htn.pth",
    map_location="cpu",
    weights_only=False
)

dataset = RWFVideoDataset(
    root, "val", checkpoint["frames"], checkpoint["size"]
)

loader = DataLoader(
    dataset, batch_size=2, shuffle=False, num_workers=0
)

model = EOHTN(frames=checkpoint["frames"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

all_probs = []
all_labels = []

with torch.no_grad():
    for videos, labels in loader:
        logits = model(videos)
        fight_probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs.extend(fight_probs.numpy())
        all_labels.extend(labels.numpy())

probs = np.array(all_probs)
labels = np.array(all_labels)

print("\n========== PROBABILITY DIAGNOSIS ==========")

for class_id, class_name in [(0, "NonFight"), (1, "Fight")]:
    values = probs[labels == class_id]

    print(f"\nActual class: {class_name}")
    print("Videos:", len(values))
    print("Minimum:", round(float(values.min()), 4))
    print("Maximum:", round(float(values.max()), 4))
    print("Mean:", round(float(values.mean()), 4))
    print("Median:", round(float(np.median(values)), 4))
    print("25th percentile:", round(float(np.percentile(values, 25)), 4))
    print("75th percentile:", round(float(np.percentile(values, 75)), 4))

print("\n========== THRESHOLD COUNTS ==========")

for threshold in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    predictions = (probs >= threshold).astype(int)
    print(
        f"Threshold {threshold:.2f}: "
        f"{int(predictions.sum())} Fight predictions, "
        f"{int((predictions == 0).sum())} NonFight predictions"
    )

print("\nDiagnosis complete. No model files were changed.")
