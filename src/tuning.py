"""Day 2, Step 2: Hyperparameter tuning with GridSearchCV.

Run from the project root:  python -m src.tuning

Tuning uses ONLY the training set (5-fold CV). The test set is touched once,
at the very end, to report the final score.
"""
import json

import joblib
import pandas as pd
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.preprocess import prepare_data
from utils.config import (BEST_MODEL_PATH, BEST_PARAMS_PATH, CV_FOLDS, RANDOM_STATE,
                          TUNING_RESULTS_PATH)
from utils.metrics import evaluate
from utils.plotting import plot_confusion_matrix, plot_grid_heatmap

PARAM_GRID = [
    {"svm__kernel": ["linear"], "svm__C": [0.01, 0.1, 1, 10, 100]},
    {"svm__kernel": ["rbf"], "svm__C": [0.1, 1, 10, 100],
     "svm__gamma": ["scale", 0.001, 0.01, 0.1, 1]},
    {"svm__kernel": ["poly"], "svm__C": [0.1, 1, 10],
     "svm__degree": [2, 3], "svm__gamma": ["scale"]},
]


def run_tuning():
    X_train, X_test, _, _, y_train, y_test, _ = prepare_data()

    pipe = Pipeline([("scaler", StandardScaler()),
                     ("svm", SVC(random_state=RANDOM_STATE))])
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"accuracy": "accuracy",
               "recall_malignant": make_scorer(recall_score, pos_label=0)}

    grid = GridSearchCV(pipe, PARAM_GRID, cv=cv, scoring=scoring,
                        refit="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    # ----- results table -----
    res = pd.DataFrame(grid.cv_results_)
    res["params_str"] = res["params"].apply(
        lambda d: ", ".join(f"{k.replace('svm__', '')}={v}" for k, v in d.items()))
    cols = ["params_str", "mean_test_accuracy", "std_test_accuracy",
            "mean_test_recall_malignant", "rank_test_accuracy"]
    res[cols].sort_values("rank_test_accuracy").to_csv(TUNING_RESULTS_PATH, index=False)

    n_fits = len(res) * CV_FOLDS
    print(f"\n--- GridSearchCV: {len(res)} combinations x {CV_FOLDS} folds = {n_fits} fits ---")
    print("\nTop 8 combinations (by CV accuracy):")
    print(res.sort_values(["rank_test_accuracy", "params_str"])[cols].head(8)
          .round(4).to_string(index=False))

    print(f"\nBest params : {grid.best_params_}")
    print(f"Best CV acc : {grid.best_score_:.4f}")

    # ----- RBF C x gamma heatmap -----
    rbf = res[res["param_svm__kernel"] == "rbf"].copy()
    rbf["gamma"] = rbf["param_svm__gamma"].astype(str)
    rbf["C"] = rbf["param_svm__C"].astype(float)
    pivot = rbf.pivot(index="C", columns="gamma", values="mean_test_accuracy")
    order = sorted([g for g in pivot.columns if g != "scale"], key=float) + ["scale"]
    plot_grid_heatmap(pivot[order], "RBF Kernel: CV Accuracy (C x gamma)", "rbf_grid_heatmap.png")

    # ----- baseline vs tuned on the held-out test set -----
    baseline = Pipeline([("scaler", StandardScaler()),
                         ("svm", SVC(kernel="linear", C=1.0, random_state=RANDOM_STATE))])
    baseline.fit(X_train, y_train)
    base_pred = baseline.predict(X_test)
    tuned_pred = grid.predict(X_test)

    base_res = evaluate(y_test, base_pred, "Baseline linear SVM (Day 1)")
    tuned_res = evaluate(y_test, tuned_pred, "Tuned SVM (Day 2)")
    mal_recall = recall_score(y_test, tuned_pred, pos_label=0)
    base_mal_recall = recall_score(y_test, base_pred, pos_label=0)
    print(f"Malignant recall (cancers caught) -> baseline: {base_mal_recall:.4f} | "
          f"tuned: {mal_recall:.4f}")

    plot_confusion_matrix(y_test, tuned_pred, "tuned_confusion_matrix.png", "Tuned SVM")

    # ----- save artefacts for Day 3 -----
    BEST_PARAMS_PATH.write_text(json.dumps(
        {k: (v if isinstance(v, (int, float, str)) else str(v))
         for k, v in grid.best_params_.items()}, indent=2))
    joblib.dump(grid.best_estimator_, BEST_MODEL_PATH)
    print(f"\n[saved] {TUNING_RESULTS_PATH.name}, {BEST_PARAMS_PATH.name}, {BEST_MODEL_PATH.name} in data/")
    return grid, base_res, tuned_res


if __name__ == "__main__":
    run_tuning()
