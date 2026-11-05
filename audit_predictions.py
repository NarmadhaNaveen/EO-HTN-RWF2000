import os
import csv
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

records = []

with torch.no_grad():
    for videos, labels in loader:
        logits = model(videos)
        probabilities = torch.softmax(logits, dim=1)

        for i in range(len(labels)):
            index = len(records)
            actual = int(labels[i])
            fight_prob = float(probabilities[i, 1])
            predicted = 1 if fight_prob >= 0.50 else 0

            records.append({
                "video": os.path.basename(dataset.samples[index][0]),
                "actual": "Fight" if actual == 1 else "NonFight",
                "predicted": "Fight" if predicted == 1 else "NonFight",
                "fight_probability": round(fight_prob, 6),
                "correct": actual == predicted
            })

os.makedirs("outputs", exist_ok=True)

with open("outputs/validation_predictions.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

false_positives = sorted(
    [r for r in records if r["actual"] == "NonFight" and not r["correct"]],
    key=lambda r: r["fight_probability"],
    reverse=True
)

false_negatives = sorted(
    [r for r in records if r["actual"] == "Fight" and not r["correct"]],
    key=lambda r: r["fight_probability"]
)

print("\n========== TOP 15 FALSE ALARMS ==========")
for r in false_positives[:15]:
    print(r["video"], "| Fight probability:", r["fight_probability"])

print("\n========== MISSED FIGHT VIDEOS ==========")
for r in false_negatives[:15]:
    print(r["video"], "| Fight probability:", r["fight_probability"])

print("\nTotal videos audited:", len(records))
print("False alarms:", len(false_positives))
print("Missed fights:", len(false_negatives))
print("\nSaved: outputs/validation_predictions.csv")
