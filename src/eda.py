"""Step 1: Exploratory Data Analysis.

Run from the project root:  python -m src.eda
"""
import pandas as pd

from utils.config import CLASS_NAMES, TARGET_COL
from utils.data_loader import load_dataset
from utils.plotting import (plot_class_distribution, plot_correlation_heatmap,
                            plot_feature_histograms)


def run_eda() -> pd.DataFrame:
    df = load_dataset()

    print("\n--- Shape ---")
    print(f"{df.shape[0]} rows, {df.shape[1]} columns (including target)")

    print("\n--- First 5 rows ---")
    print(df.head())

    print("\n--- Missing values ---")
    print(f"Total missing: {int(df.isna().sum().sum())}")

    print("\n--- Duplicate rows ---")
    print(f"Duplicates: {int(df.duplicated().sum())}")

    print("\n--- Class balance ---")
    counts = df[TARGET_COL].value_counts()
    for cls, n in counts.items():
        print(f"{CLASS_NAMES[cls]:>10}: {n} ({n / len(df):.1%})")

    print("\n--- Summary statistics (first 6 features) ---")
    print(df.iloc[:, :6].describe().round(2))

    print("\n--- Top 5 features correlated with target ---")
    corr = df.corr()[TARGET_COL].drop(TARGET_COL).sort_values(key=abs, ascending=False)
    print(corr.head(5).round(3))

    plot_class_distribution(df)
    plot_feature_histograms(df)
    plot_correlation_heatmap(df)
    return df


if __name__ == "__main__":
    run_eda()
