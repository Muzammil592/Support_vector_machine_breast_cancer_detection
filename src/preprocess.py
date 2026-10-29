"""Step 2: Train/test split and feature scaling.

Key rule: fit the scaler on TRAINING data only, then apply it to the test data.
Fitting on all data leaks information from the test set (data leakage).
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.config import RANDOM_STATE, TEST_SIZE
from utils.data_loader import get_features_and_target, load_dataset
from utils.plotting import plot_scale_comparison


def split_data(X, y):
    """80/20 stratified split so both sets keep the same class ratio."""
    return train_test_split(X, y, test_size=TEST_SIZE,
                            stratify=y, random_state=RANDOM_STATE)


def scale_data(X_train, X_test):
    """Fit StandardScaler on train only; transform both sets."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train),
                                  columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test),
                                 columns=X_test.columns, index=X_test.index)
    return X_train_scaled, X_test_scaled, scaler


def prepare_data(make_plot: bool = False):
    """Load -> split -> scale. Returns raw and scaled versions."""
    df = load_dataset()
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_s, X_test_s, scaler = scale_data(X_train, X_test)

    print(f"[split] Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"[split] Train class ratio (benign): {y_train.mean():.3f} | "
          f"Test: {y_test.mean():.3f}")

    if make_plot:
        plot_scale_comparison(X_train, X_train_s)

    return X_train, X_test, X_train_s, X_test_s, y_train, y_test, scaler


if __name__ == "__main__":
    prepare_data(make_plot=True)
