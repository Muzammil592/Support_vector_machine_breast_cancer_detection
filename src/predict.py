"""Day 3 bonus: predict with the saved model.

Examples (from the project root):
    python -m src.predict --index 0 5 12      # rows from the dataset
    python -m src.predict --csv my_samples.csv   # CSV with the same 30 feature columns

Educational project only: this is NOT a medical device or diagnostic tool.
"""
import argparse

import pandas as pd

from src.evaluate import load_best_model
from utils.config import CLASS_NAMES, TARGET_COL
from utils.data_loader import load_dataset
from utils.metrics import malignant_scores


def main():
    parser = argparse.ArgumentParser(description="Predict malignant/benign with the saved SVM.")
    parser.add_argument("--index", type=int, nargs="+", help="Row numbers from the dataset")
    parser.add_argument("--csv", type=str, help="CSV file with the 30 feature columns")
    args = parser.parse_args()

    df = load_dataset()
    model = load_best_model(df.drop(columns=[TARGET_COL]), df[TARGET_COL])

    if args.csv:
        X = pd.read_csv(args.csv)
        truth = X[TARGET_COL].map(CLASS_NAMES) if TARGET_COL in X.columns else None
        X = X.drop(columns=[TARGET_COL], errors="ignore")
    else:
        idx = args.index or [0, 1, 2]
        X = df.drop(columns=[TARGET_COL]).iloc[idx]
        truth = df[TARGET_COL].iloc[idx].map(CLASS_NAMES)

    out = pd.DataFrame({
        "prediction": pd.Series(model.predict(X), index=X.index).map(CLASS_NAMES),
        "malignant_score": malignant_scores(model, X).round(3)}, index=X.index)
    if truth is not None:
        out.insert(0, "true_label", truth)
    print(out.to_string())
    print("\nmalignant_score > 0 means predicted malignant. Higher = more confident.")
    print("Educational project only. Not for medical use.")


if __name__ == "__main__":
    main()
