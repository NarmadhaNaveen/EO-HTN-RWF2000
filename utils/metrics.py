import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

def expected_calibration_error(probs, labels, bins=10):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)

    for i in range(bins):
        if i == bins - 1:
            mask = (confidences >= edges[i]) & (confidences <= edges[i+1])
        else:
            mask = (confidences >= edges[i]) & (confidences < edges[i+1])

        if mask.any():
            accuracy = (predictions[mask] == labels[mask]).mean()
            confidence = confidences[mask].mean()
            ece += mask.mean() * abs(accuracy - confidence)

    return float(ece)


def calculate_metrics(labels, probs):
    predictions = probs.argmax(axis=1)
    one_hot = np.eye(2)[labels]

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "brier": np.mean(np.sum((probs - one_hot) ** 2, axis=1)),
        "ece": expected_calibration_error(probs, labels),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist()
    }
