"""Evaluation helpers."""
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from utils.config import CLASS_NAMES


def evaluate(y_true, y_pred, label: str = "Model") -> dict:
    """Print a short report and return the main metrics as a dict."""
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }
    print(f"\n===== {label} =====")
    for k, v in results.items():
        print(f"{k:>10}: {v:.4f}")
    print(classification_report(y_true, y_pred,
                                target_names=[CLASS_NAMES[0], CLASS_NAMES[1]]))
    return results
