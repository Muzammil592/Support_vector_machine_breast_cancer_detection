"""Functions for loading the Breast Cancer Wisconsin dataset."""
import pandas as pd
from sklearn.datasets import load_breast_cancer

from utils.config import DATA_DIR, DATASET_PATH, TARGET_COL


def download_dataset() -> pd.DataFrame:
    """Load the dataset from scikit-learn and save it as CSV in data/."""
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_PATH, index=False)
    print(f"[data] Dataset saved to {DATASET_PATH}")
    return df


def load_dataset() -> pd.DataFrame:
    """Return the dataset as a DataFrame. Creates the CSV if it is missing."""
    if not DATASET_PATH.exists():
        return download_dataset()
    return pd.read_csv(DATASET_PATH)


def get_features_and_target(df: pd.DataFrame):
    """Split a DataFrame into features X and target y."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y
