import os
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.eo_htn import EOHTN
from utils.video_data import RWFVideoDataset
from utils.metrics import calculate_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--size", type=int, default=96)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)

    root = os.path.join(os.path.dirname(__file__), "RWF-2000")
    train_ds = RWFVideoDataset(root, "train", args.frames, args.size)
    val_ds = RWFVideoDataset(root, "val", args.frames, args.size)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=0
    )

    model = EOHTN(frames=args.frames)
    device = torch.device("cpu")
    model.to(device)

    print("Device:", device)
    print("Trainable parameters:",
          sum(p.numel() for p in model.parameters() if p.requires_grad))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0003, weight_decay=0.0001
    )

    best_f1 = -1.0
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for step, (videos, labels) in enumerate(train_loader):
            videos, labels = videos.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(videos)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (step + 1) % 50 == 0:
                print(f"Epoch {epoch+1} | Batch {step+1}/{len(train_loader)} "
                      f"| Loss {loss.item():.4f}")

        model.eval()
        all_probs, all_labels = [], []

        with torch.no_grad():
            for videos, labels in val_loader:
                logits = model(videos.to(device))
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.numpy())

        probs = np.concatenate(all_probs)
        labels = np.concatenate(all_labels)
        metrics = calculate_metrics(labels, probs)
        average_loss = total_loss / max(len(train_loader), 1)

        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Training loss: {average_loss:.4f}")
        for key in ["accuracy", "precision", "recall", "f1", "brier", "ece"]:
            print(f"{key}: {metrics[key]:.4f}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "frames": args.frames,
                "size": args.size,
                "metrics": metrics
            }, "checkpoints/best_eo_htn.pth")
            print("Best model checkpoint saved.")

        with open("logs/training_metrics.json", "w") as f:
            json.dump({"epoch": epoch+1, "metrics": metrics}, f, indent=2)

    print("\nTraining finished.")
    print("Best validation F1:", best_f1)
    print("Checkpoint: checkpoints/best_eo_htn.pth")

if __name__ == "__main__":
    main()
