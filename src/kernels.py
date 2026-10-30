"""Day 2, Step 1: Compare SVM kernels with cross-validation.

Run from the project root:  python -m src.kernels

The scaler lives INSIDE the pipeline, so in every CV fold it is fitted on that
fold's training part only (no leakage into the validation part).
"""
import time

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.preprocess import prepare_data
from utils.config import CV_FOLDS, RANDOM_STATE
from utils.plotting import plot_kernel_comparison

KERNELS = ["linear", "poly", "rbf", "sigmoid"]


def run_kernel_comparison() -> pd.DataFrame:
    X_train, _, _, _, y_train, _, _ = prepare_data()
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for kernel in KERNELS:
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("svm", SVC(kernel=kernel, random_state=RANDOM_STATE))])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
        start = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_time = time.perf_counter() - start
        rows.append({
            "kernel": kernel,
            "cv_mean": scores.mean(),
            "cv_std": scores.std(),
            "train_acc": pipe.score(X_train, y_train),
            "n_support": int(pipe.named_steps["svm"].n_support_.sum()),
            "fit_time_s": fit_time,
        })

    df = pd.DataFrame(rows)
    print("\n--- Kernel comparison (5-fold CV on training set, default C=1) ---")
    print(df.round(4).to_string(index=False))
    plot_kernel_comparison(df)
    return df


if __name__ == "__main__":
    run_kernel_comparison()
