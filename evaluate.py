import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.eo_htn import EOHTN
from utils.video_data import RWFVideoDataset
from utils.metrics import calculate_metrics


def threshold_metrics(labels, fight_probs, threshold):
    predictions = (fight_probs >= threshold).astype(int)

    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tp = int(np.sum((labels == 1) & (predictions == 1)))

    accuracy = (tp + tn) / max(len(labels), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_negative": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positive": tp
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["train", "val"])
    args = parser.parse_args()

    torch.set_num_threads(2)

    root = os.path.join(os.path.dirname(__file__), "RWF-2000")

    checkpoint = torch.load(
        "checkpoints/best_eo_htn.pth",
        map_location="cpu",
        weights_only=False
    )

    dataset = RWFVideoDataset(
        root, args.split, checkpoint["frames"], checkpoint["size"]
    )

    loader = DataLoader(
        dataset, batch_size=2, shuffle=False, num_workers=0
    )

    model = EOHTN(frames=checkpoint["frames"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    probs_list, labels_list = [], []

    with torch.no_grad():
        for videos, labels in loader:
            logits = model(videos)
            probabilities = torch.softmax(logits, dim=1)

            probs_list.append(probabilities.numpy())
            labels_list.append(labels.numpy())

    probs = np.concatenate(probs_list)
    labels = np.concatenate(labels_list)

    # Original evaluation using the default 0.50 decision threshold
    original_results = calculate_metrics(labels, probs)

    print("\nORIGINAL EVALUATION")
    print(json.dumps(original_results, indent=2))

    # Compare different Fight decision thresholds
    fight_probs = probs[:, 1]

    thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    threshold_results = []

    for threshold in thresholds:
        result = threshold_metrics(labels, fight_probs, threshold)
        threshold_results.append(result)

    print("\nTHRESHOLD COMPARISON")
    print(
        f"{'Threshold':<12}"
        f"{'Accuracy':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'False Alarms':<15}"
        f"{'Missed Fights':<15}"
    )

    for result in threshold_results:
        print(
            f"{result['threshold']:<12.2f}"
            f"{result['accuracy']:<12.4f}"
            f"{result['precision']:<12.4f}"
            f"{result['recall']:<12.4f}"
            f"{result['f1']:<12.4f}"
            f"{result['false_positives']:<15}"
            f"{result['false_negatives']:<15}"
        )

    os.makedirs("outputs", exist_ok=True)

    with open(f"outputs/{args.split}_evaluation.json", "w") as f:
        json.dump(original_results, f, indent=2)

    with open(f"outputs/{args.split}_threshold_analysis.json", "w") as f:
        json.dump(threshold_results, f, indent=2)

    print("\nSaved evaluation and threshold analysis in outputs/")


if __name__ == "__main__":
    main()
