import os
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, PROJECT_ROOT)

from torch.utils.data import DataLoader
from models.eo_htn import EOHTN
from utils.video_data_v2 import RWFVideoDataset
from utils.metrics import calculate_metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    set_seed(42)
    torch.set_num_threads(2)

    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    root = os.path.join(project_root, "RWF-2000")
    output_root = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(output_root, "checkpoints")
    log_dir = os.path.join(output_root, "logs")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_ds = RWFVideoDataset(
        root, "train", args.frames, args.size
    )
    val_ds = RWFVideoDataset(
        root, "val", args.frames, args.size
    )

    generator = torch.Generator()
    generator.manual_seed(42)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    device = torch.device("cpu")
    model = EOHTN(frames=args.frames).to(device)

    parameter_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print("Device:", device)
    print("Trainable parameters:", parameter_count)
    print("Training videos:", len(train_ds))
    print("Validation videos:", len(val_ds))

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0003,
        weight_decay=0.0001
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    best_f1 = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for step, (videos, labels) in enumerate(train_loader):
            videos = videos.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(videos)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            predictions = logits.argmax(dim=1)
            correct_train += (predictions == labels).sum().item()
            total_train += labels.size(0)

            if (step + 1) % 50 == 0:
                print(
                    f"Epoch {epoch+1} | Batch {step+1}/"
                    f"{len(train_loader)} | Loss {loss.item():.4f}"
                )

        average_train_loss = (
            total_train_loss / max(len(train_loader), 1)
        )
        train_accuracy = correct_train / max(total_train, 1)

        model.eval()
        total_val_loss = 0.0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for videos, labels in val_loader:
                videos = videos.to(device)
                labels = labels.to(device)

                logits = model(videos)
                loss = criterion(logits, labels)
                probabilities = torch.softmax(logits, dim=1)

                total_val_loss += loss.item()
                all_probs.append(probabilities.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        probs = np.concatenate(all_probs)
        labels = np.concatenate(all_labels)

        metrics = calculate_metrics(labels, probs)
        average_val_loss = (
            total_val_loss / max(len(val_loader), 1)
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(metrics["f1"])

        record = {
            "epoch": epoch + 1,
            "learning_rate": current_lr,
            "train_loss": average_train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": average_val_loss,
            "validation_metrics": metrics
        }

        history.append(record)

        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Learning rate: {current_lr:.7f}")
        print(f"Training loss: {average_train_loss:.4f}")
        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Validation loss: {average_val_loss:.4f}")

        for key in [
            "accuracy", "precision", "recall",
            "f1", "brier", "ece"
        ]:
            print(f"Validation {key}: {metrics[key]:.4f}")

        improved = metrics["f1"] > best_f1

        if improved:
            best_f1 = metrics["f1"]
            epochs_without_improvement = 0

            checkpoint_path = os.path.join(
                checkpoint_dir, "best_eo_htn_v2.pth"
            )

            torch.save({
                "model_state_dict": model.state_dict(),
                "frames": args.frames,
                "size": args.size,
                "epoch": epoch + 1,
                "metrics": metrics,
                "parameters": parameter_count
            }, checkpoint_path)

            print("Best experimental checkpoint saved.")
        else:
            epochs_without_improvement += 1

        log_path = os.path.join(log_dir, "training_history.json")

        with open(log_path, "w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)

        if epochs_without_improvement >= args.patience:
            print("\nEarly stopping triggered.")
            break

    print("\nTraining finished.")
    print("Best validation F1:", best_f1)
    print("History:", log_path)
    print(
        "Checkpoint:",
        os.path.join(checkpoint_dir, "best_eo_htn_v2.pth")
    )


if __name__ == "__main__":
    main()
