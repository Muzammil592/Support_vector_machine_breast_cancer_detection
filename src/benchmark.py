"""Day 3, Step 2: Benchmark SVM against Logistic Regression and Random Forest.

Run from the project root:  python -m src.benchmark

Two views are produced:
  A) Hold-out test: each model is tuned on the training set, then scored once on the test set.
  B) Nested cross-validation on ALL 569 samples (inner loop tunes, outer loop scores).
     This is the more reliable comparison, because a 114-sample test set is small.
"""
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.preprocess import prepare_data
from src.tuning import PARAM_GRID as SVM_GRID
from utils.config import (BENCHMARK_RESULTS_PATH, CV_FOLDS, NESTED_REPEATS, RANDOM_STATE)
from utils.data_loader import get_features_and_target, load_dataset
from utils.metrics import malignant_scores
from utils.plotting import plot_benchmark, plot_pr_curves, plot_roc_curves


def build_models() -> dict:
    """name -> (pipeline, parameter grid)."""
    return {
        "SVM": (
            Pipeline([("scaler", StandardScaler()), ("svm", SVC(random_state=RANDOM_STATE))]),
            SVM_GRID),
        "Logistic Regression": (
            Pipeline([("scaler", StandardScaler()),
                      ("lr", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))]),
            {"lr__C": [0.01, 0.1, 1, 10, 100]}),
        "Random Forest": (
            Pipeline([("rf", RandomForestClassifier(random_state=RANDOM_STATE))]),
            {"rf__n_estimators": [200], "rf__max_depth": [None, 5, 10],
             "rf__min_samples_leaf": [1, 3]}),
    }


def holdout_comparison(models: dict):
    X_train, X_test, _, _, y_train, y_test, _ = prepare_data()
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows, curves = [], []
    for name, (pipe, grid) in models.items():
        search = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=-1)
        t0 = time.perf_counter()
        search.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        pred = search.predict(X_test)
        scores = malignant_scores(search.best_estimator_, X_test)
        rows.append({
            "model": name,
            "best_params": ", ".join(f"{k.split('__')[1]}={v}" for k, v in search.best_params_.items()),
            "cv_accuracy": search.best_score_,
            "test_accuracy": accuracy_score(y_test, pred),
            "test_recall_malignant": recall_score(y_test, pred, pos_label=0),
            "test_precision_malignant": precision_score(y_test, pred, pos_label=0),
            "test_roc_auc": roc_auc_score(y_test == 0, scores),
            "tuning_time_s": fit_time,
        })
        curves.append((name, y_test, scores))
    df = pd.DataFrame(rows)
    print("\n--- A) Hold-out test comparison (tuned on train, scored on test) ---")
    print(df.round(4).to_string(index=False))
    plot_roc_curves(curves, "benchmark_roc.png", title="ROC: model comparison (test set)")
    plot_pr_curves(curves, "benchmark_pr.png", title="Precision-Recall: model comparison (test set)")
    return df


def nested_cv_comparison(models: dict) -> pd.DataFrame:
    X, y = get_features_and_target(load_dataset())
    inner = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=NESTED_REPEATS,
                                    random_state=RANDOM_STATE)
    scoring = {"accuracy": "accuracy",
               "recall_malignant": make_scorer(recall_score, pos_label=0),
               "roc_auc": "roc_auc"}
    print(f"\n[nested CV] 5 outer folds x {NESTED_REPEATS} repeats; inner {CV_FOLDS}-fold tuning...")

    rows = []
    for name, (pipe, grid) in models.items():
        search = GridSearchCV(pipe, grid, cv=inner, scoring="accuracy", n_jobs=1)
        res = cross_validate(search, X, y, cv=outer, scoring=scoring, n_jobs=-1)
        row = {"model": name}
        for m in scoring:
            row[f"{m}_mean"] = res[f"test_{m}"].mean()
            row[f"{m}_std"] = res[f"test_{m}"].std()
        rows.append(row)
        print(f"  done: {name}")
    df = pd.DataFrame(rows)
    print("\n--- B) Nested CV on all 569 samples (mean +/- std over outer folds) ---")
    print(df.round(4).to_string(index=False))
    plot_benchmark(df)
    return df


def run_benchmark() -> pd.DataFrame:
    models = build_models()
    holdout = holdout_comparison(models)
    nested = nested_cv_comparison(models)
    merged = holdout.merge(nested, on="model")
    merged.to_csv(BENCHMARK_RESULTS_PATH, index=False)
    print(f"\n[saved] {BENCHMARK_RESULTS_PATH.name} in data/")
    return merged


if __name__ == "__main__":
    run_benchmark()
