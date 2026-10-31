"""Day 3, Step 1: Final evaluation of the tuned SVM.

Run from the project root:  python -m src.evaluate

Malignant is the POSITIVE class here. The test set is used for reporting only:
the decision threshold is chosen on out-of-fold scores from the TRAINING set.
"""
import json

import joblib
from sklearn.base import clone
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.preprocess import prepare_data
from utils.config import (BEST_MODEL_PATH, CV_FOLDS, FINAL_RESULTS_PATH, N_BOOTSTRAP,
                          RANDOM_STATE, TARGET_RECALL)
from utils.metrics import (bootstrap_ci, default_threshold, evaluate, malignant_scores,
                           operating_point, pick_threshold)
from utils.plotting import (plot_confusion_matrix, plot_pr_curves, plot_roc_curves,
                            plot_threshold_tradeoff)


def load_best_model(X_train, y_train):
    """Load the Day 2 model, or run Day 2 tuning if it has not been saved yet."""
    if BEST_MODEL_PATH.exists():
        print(f"[model] Loaded {BEST_MODEL_PATH.name}")
        return joblib.load(BEST_MODEL_PATH)
    print("[model] No saved model found, running Day 2 tuning first...")
    from src.tuning import run_tuning
    grid, _, _ = run_tuning()
    return grid.best_estimator_


def _print_point(label: str, p: dict) -> None:
    print(f"{label:<28} recall={p['recall']:.3f}  precision={p['precision']:.3f}  "
          f"caught={p['caught']}  missed={p['missed']}  false alarms={p['false_alarms']}")


def run_evaluation() -> dict:
    X_train, X_test, _, _, y_train, y_test, _ = prepare_data()
    model = load_best_model(X_train, y_train)
    print(f"[model] {model.named_steps['svm']}")

    # ---------- 1. Standard test metrics, malignant = positive ----------
    y_pred = model.predict(X_test)
    scores_test = malignant_scores(model, X_test)
    metrics = evaluate(y_test, y_pred, "Tuned SVM on test set", pos_label=0)
    pos = (y_test == 0).to_numpy()
    auc = roc_auc_score(pos, scores_test)
    ap = average_precision_score(pos, scores_test)
    print(f"ROC AUC: {auc:.4f} | Average precision: {ap:.4f}")

    t0 = default_threshold(model)
    default_pt = operating_point(y_test, scores_test, t0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print(f"\nConfusion matrix (rows = true, cols = predicted; order malignant, benign):\n{cm}")
    plot_confusion_matrix(y_test, y_pred, "final_confusion_matrix.png",
                          "Final Tuned SVM (test set)")

    # ---------- 2. Bootstrap confidence intervals ----------
    ci = bootstrap_ci(y_test, y_pred, scores_test, N_BOOTSTRAP, RANDOM_STATE)
    print(f"\n95% bootstrap confidence intervals ({N_BOOTSTRAP} resamples of the test set):")
    for k, (lo, hi) in ci.items():
        print(f"  {k:<20} [{lo:.3f}, {hi:.3f}]")

    # ---------- 3. Threshold tuning on TRAINING out-of-fold scores ----------
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    method = "decision_function" if hasattr(model, "decision_function") else "predict_proba"
    oof = cross_val_predict(clone(model), X_train, y_train, cv=cv, method=method)
    oof_mal = -oof if method == "decision_function" else oof[:, 0]

    t_chosen = pick_threshold(oof_mal, y_train, TARGET_RECALL)
    oof_default = operating_point(y_train, oof_mal, t0)
    oof_chosen = operating_point(y_train, oof_mal, t_chosen)
    test_chosen = operating_point(y_test, scores_test, t_chosen)

    print(f"\n--- Threshold tuning (target malignant recall >= {TARGET_RECALL:.0%} on training CV) ---")
    print(f"Chosen threshold: {t_chosen:.3f} (default {t0:.3f})")
    _print_point("Train OOF, default", oof_default)
    _print_point("Train OOF, chosen", oof_chosen)
    _print_point("TEST, default", default_pt)
    _print_point("TEST, chosen", test_chosen)

    # ---------- 4. Plots ----------
    curves = [("Tuned SVM", y_test, scores_test)]
    marks = [(default_pt["false_positive_rate"], default_pt["recall"], "Default threshold"),
             (test_chosen["false_positive_rate"], test_chosen["recall"], "Chosen threshold")]
    plot_roc_curves(curves, "final_roc_curve.png", marks=marks)
    plot_pr_curves(curves, "final_pr_curve.png")
    plot_threshold_tradeoff(oof_mal, y_train, t0, t_chosen)

    results = {
        "model": str(model.named_steps["svm"]),
        "test_metrics_malignant_positive": {k: float(v) for k, v in metrics.items()},
        "roc_auc": float(auc), "average_precision": float(ap),
        "confusion_matrix": cm.tolist(),
        "bootstrap_95ci": ci,
        "threshold": {"target_recall": TARGET_RECALL, "default": t0, "chosen": t_chosen,
                      "train_oof_default": oof_default, "train_oof_chosen": oof_chosen,
                      "test_default": default_pt, "test_chosen": test_chosen},
    }
    FINAL_RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n[saved] {FINAL_RESULTS_PATH.name} in data/")
    return results


if __name__ == "__main__":
    run_evaluation()
