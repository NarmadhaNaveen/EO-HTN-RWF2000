import os
import csv
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(
    BASE_DIR, "logs", "v2_validation_predictions.csv"
)
OUTPUT_FILE = os.path.join(
    BASE_DIR, "logs", "v2_threshold_analysis.csv"
)

if not os.path.isfile(INPUT_FILE):
    raise FileNotFoundError(INPUT_FILE)

actual = []
probabilities = []

with open(INPUT_FILE, "r", encoding="utf-8", newline="") as file:
    reader = csv.DictReader(file)
    for row in reader:
        actual.append(1 if row["actual_label"] == "Fight" else 0)
        probabilities.append(float(row["probability_fight"]))

y_true = np.asarray(actual, dtype=int)
y_prob = np.asarray(probabilities, dtype=float)

if len(y_true) == 0:
    raise RuntimeError("No predictions found.")

if not np.isfinite(y_prob).all():
    raise RuntimeError("Invalid probability values found.")

thresholds = np.round(np.arange(0.30, 0.801, 0.05), 2)
results = []

print("\n========== V2 THRESHOLD ANALYSIS ==========")
print("Videos:", len(y_true))
print("Positive class: Fight")
print("Prediction rule: Fight probability >= threshold")
print()

for threshold in thresholds:
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, zero_division=0
    )
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) else 0
    fpr = fp / (fp + tn) if (fp + tn) else 0

    results.append({
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "false_positive_rate": fpr,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    })

    print(
        f"{threshold:.2f} | "
        f"Acc {accuracy:.4f} | "
        f"Prec {precision:.4f} | "
        f"Recall {recall:.4f} | "
        f"F1 {f1:.4f} | "
        f"Specificity {specificity:.4f} | "
        f"FP {fp} | FN {fn}"
    )

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

best_f1 = max(results, key=lambda x: x["f1"])
best_balance = max(
    results,
    key=lambda x: (x["specificity"] + x["recall"]) / 2
)

print("\n========== VALIDATION ANALYSIS ==========")
print("Best F1 threshold:", best_f1["threshold"])
print("F1:", round(best_f1["f1"], 4))
print("Precision:", round(best_f1["precision"], 4))
print("Recall:", round(best_f1["recall"], 4))
print("False positives:", best_f1["FP"])
print("False negatives:", best_f1["FN"])

print("\nBest recall-specificity balance in this grid:")
print("Threshold:", best_balance["threshold"])
print("Recall:", round(best_balance["recall"], 4))
print("Specificity:", round(best_balance["specificity"], 4))
print("F1:", round(best_balance["f1"], 4))

print("\nSaved results:", OUTPUT_FILE)
print("\nNOTE: Thresholds are being explored on validation data.")
print("Do not report the selected threshold as independent test performance.")
