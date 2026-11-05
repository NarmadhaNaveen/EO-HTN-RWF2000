import os
import sys
import csv
import json
import torch
import numpy as np
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, PROJECT_ROOT)

from models.eo_htn import EOHTN
from utils.video_data_v2 import RWFVideoDataset
from utils.metrics import calculate_metrics

DEVICE = torch.device("cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = PROJECT_ROOT
DATASET_ROOT = os.path.join(PROJECT_DIR, "RWF-2000")

CHECKPOINT_PATH = os.path.join(
    BASE_DIR, "checkpoints", "best_eo_htn_v2.pth"
)
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(LOG_DIR, exist_ok=True)

def main():
    print("\n========== EO-HTN V2 CHECKPOINT EVALUATION ==========")

    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(CHECKPOINT_PATH)

    # Load the saved V2 checkpoint.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    frames = int(checkpoint["frames"])
    size = int(checkpoint["size"])
    saved_epoch = int(checkpoint["epoch"])

    print("Checkpoint:", CHECKPOINT_PATH)
    print("Checkpoint epoch:", saved_epoch)
    print("Frames:", frames)
    print("Image size:", size)
    print("Device:", DEVICE)

    # Build the same model architecture used during V2 training.
    model = EOHTN(frames=frames).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    parameter_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print("Trainable parameters:", parameter_count)

    # Validation dataset: centre sampling, no augmentation.
    val_dataset = RWFVideoDataset(
        DATASET_ROOT,
        split="val",
        frames=frames,
        size=size
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )

    all_labels = []
    all_probabilities = []
    all_predictions = []

    print("\nEvaluating validation videos...")

    with torch.inference_mode():
        for batch_index, (videos, labels) in enumerate(val_loader):
            videos = videos.to(DEVICE)

            logits = model(videos)
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)

            all_labels.extend(labels.numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())

            if (batch_index + 1) % 50 == 0:
                print(
                    f"Processed {(batch_index + 1) * 2}/"
                    f"{len(val_dataset)} videos"
                )

    labels = np.asarray(all_labels, dtype=np.int64)
    probabilities = np.asarray(all_probabilities, dtype=np.float64)
    predictions = np.asarray(all_predictions, dtype=np.int64)

    if len(labels) != len(val_dataset):
        raise RuntimeError("Evaluation count does not match dataset size.")

    # Use the same metric function as the training script.
    metrics = calculate_metrics(labels, probabilities)

    print("\n========== EVALUATION RESULTS ==========")
    print("Checkpoint epoch:", saved_epoch)
    print("Validation videos:", len(labels))

    for key in ["accuracy", "precision", "recall", "f1", "brier", "ece"]:
        print(f"{key}: {metrics[key]:.6f} ({metrics[key] * 100:.2f}%)"
              if key in ["accuracy", "precision", "recall", "f1"]
              else f"{key}: {metrics[key]:.6f}")

    cm = np.asarray(metrics["confusion_matrix"], dtype=int)

    print("\nConfusion matrix (rows=actual, columns=predicted)")
    print("Class order: NonFight=0, Fight=1")
    print(cm)

    tn, fp, fn, tp = cm.ravel()

    print("\n========== CLASSIFICATION COUNTS ==========")
    print("True NonFight:", tn)
    print("False Fight alarms:", fp)
    print("Missed Fight videos:", fn)
    print("Correct Fight detections:", tp)

    # Save predictions in the same order as the validation dataset.
    prediction_path = os.path.join(LOG_DIR, "v2_validation_predictions.csv")

    with open(prediction_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "video_path",
            "actual_label",
            "predicted_label",
            "probability_nonfight",
            "probability_fight",
            "correct"
        ])

        for index, (video_path, actual) in enumerate(val_dataset.samples):
            predicted = int(predictions[index])
            writer.writerow([
                video_path,
                "Fight" if actual == 1 else "NonFight",
                "Fight" if predicted == 1 else "NonFight",
                float(probabilities[index][0]),
                float(probabilities[index][1]),
                actual == predicted
            ])

    # Save evaluation metrics.
    metrics_output = {
        "checkpoint_epoch": saved_epoch,
        "checkpoint_path": CHECKPOINT_PATH,
        "evaluation_split": "validation",
        "validation_videos": len(labels),
        "frames": frames,
        "image_size": size,
        "trainable_parameters": parameter_count,
        "metrics": {
            key: (
                np.asarray(value).tolist()
                if isinstance(value, np.ndarray)
                else value
            )
            for key, value in metrics.items()
        }
    }

    metrics_path = os.path.join(LOG_DIR, "v2_checkpoint_evaluation.json")

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics_output, file, indent=2)

    # Save confusion matrix image.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(cm)

        ax.set_title(f"EO-HTN V2 Validation Confusion Matrix - Epoch {saved_epoch}")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["NonFight", "Fight"])
        ax.set_yticklabels(["NonFight", "Fight"])

        for row in range(2):
            for col in range(2):
                ax.text(col, row, str(cm[row, col]),
                        ha="center", va="center")

        fig.tight_layout()

        matrix_path = os.path.join(LOG_DIR, "v2_confusion_matrix.png")
        fig.savefig(matrix_path, dpi=160)
        plt.close(fig)

        print("\nConfusion matrix image:", matrix_path)

    except ImportError:
        print("\nMatplotlib unavailable; confusion matrix image not created.")

    print("\n========== SAVED OUTPUTS ==========")
    print("Predictions CSV:", prediction_path)
    print("Metrics JSON:", metrics_path)
    print("\nEvaluation completed successfully.")

if __name__ == "__main__":
    main()
