"""Evaluation helpers."""
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from utils.config import CLASS_NAMES


def evaluate(y_true, y_pred, label: str = "Model", pos_label: int = 1) -> dict:
    """Print a short report and return the main metrics as a dict.

    pos_label=1 treats benign as the positive class (Day 1 convention).
    pos_label=0 treats malignant as positive, so recall = share of cancers caught.
    """
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=pos_label),
        "recall": recall_score(y_true, y_pred, pos_label=pos_label),
        "f1": f1_score(y_true, y_pred, pos_label=pos_label),
    }
    print(f"\n===== {label} (positive class: {CLASS_NAMES[pos_label]}) =====")
    for k, v in results.items():
        print(f"{k:>10}: {v:.4f}")
    print(classification_report(y_true, y_pred,
                                target_names=[CLASS_NAMES[0], CLASS_NAMES[1]]))
    return results
