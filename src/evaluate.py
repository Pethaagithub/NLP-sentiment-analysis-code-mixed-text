"""
Evaluation utilities: precision / recall / F1 (macro) / accuracy, plus a
confusion matrix plot, matching the metrics reported in the original study.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return {
        "precision_macro": round(float(precision), 4),
        "recall_macro": round(float(recall), 4),
        "f1_macro": round(float(f1), 4),
        "accuracy": round(float(accuracy), 4),
    }


def save_metrics(metrics: dict, run_dir: str, filename: str = "metrics.json"):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, filename), "w") as f:
        json.dump(metrics, f, indent=2)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list, run_dir: str,
                           filename: str = "confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    os.makedirs(run_dir, exist_ok=True)
    plt.savefig(os.path.join(run_dir, filename))
    plt.close()
