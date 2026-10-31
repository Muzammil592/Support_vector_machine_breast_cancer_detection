"""Evaluation helpers.

Convention from Day 3 onwards: MALIGNANT (class 0) is the positive class, because
missing a cancer is the costly error. A "malignant score" is higher when the model
thinks a sample is more likely malignant.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)

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


def malignant_scores(model, X) -> np.ndarray:
    """Return a score that is HIGHER when the sample looks more malignant.

    Works for models with decision_function (SVM, Logistic Regression), where a
    positive value means class 1 (benign), and for models with predict_proba only.
    """
    if hasattr(model, "decision_function"):
        return -np.asarray(model.decision_function(X))
    return np.asarray(model.predict_proba(X))[:, 0]


def default_threshold(model) -> float:
    """Natural decision threshold on the malignant score."""
    return 0.0 if hasattr(model, "decision_function") else 0.5


def operating_point(y_true, scores, threshold: float) -> dict:
    """Confusion counts and rates when 'malignant' is predicted if score >= threshold."""
    y_mal = np.asarray(y_true) == 0
    pred = np.asarray(scores) >= threshold
    tp = int((pred & y_mal).sum())
    fn = int((~pred & y_mal).sum())
    fp = int((pred & ~y_mal).sum())
    tn = int((~pred & ~y_mal).sum())
    return {
        "threshold": float(threshold),
        "caught": tp, "missed": fn, "false_alarms": fp, "true_benign": tn,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "accuracy": (tp + tn) / len(y_mal),
    }


def pick_threshold(scores, y_true, target_recall: float) -> float:
    """Largest threshold whose recall on (out-of-fold) data reaches target_recall.

    The largest qualifying threshold gives the fewest false alarms for that recall.
    """
    y_mal = np.asarray(y_true) == 0
    scores = np.asarray(scores)
    for t in np.sort(np.unique(scores))[::-1]:
        if (scores[y_mal] >= t).mean() >= target_recall:
            return float(t)
    return float(scores.min())


def bootstrap_ci(y_true, y_pred, scores, n_boot: int, seed: int) -> dict:
    """95% bootstrap confidence intervals for the main test metrics."""
    rng = np.random.default_rng(seed)
    y_true, y_pred, scores = map(np.asarray, (y_true, y_pred, scores))
    n = len(y_true)
    stats = {"accuracy": [], "recall_malignant": [], "precision_malignant": [], "roc_auc": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp, sc = y_true[idx], y_pred[idx], scores[idx]
        if len(np.unique(yt)) < 2:
            continue
        stats["accuracy"].append(accuracy_score(yt, yp))
        stats["recall_malignant"].append(recall_score(yt, yp, pos_label=0))
        stats["precision_malignant"].append(precision_score(yt, yp, pos_label=0, zero_division=0))
        stats["roc_auc"].append(roc_auc_score(yt == 0, sc))
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for k, v in stats.items()}
