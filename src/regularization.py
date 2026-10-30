"""Day 2, Step 3: Experiment with C (very low vs very high).

Run from the project root:  python -m src.regularization
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.preprocess import prepare_data
from utils.config import CV_FOLDS, RANDOM_STATE
from utils.plotting import plot_c_sweep

C_VALUES = np.logspace(-3, 3, 13)


def run_c_sweep() -> pd.DataFrame:
    X_train, _, _, _, y_train, _, _ = prepare_data()
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("svm", SVC(kernel="rbf", gamma="scale", random_state=RANDOM_STATE))])
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    train_scores, cv_scores = validation_curve(
        pipe, X_train, y_train, param_name="svm__C", param_range=C_VALUES,
        cv=cv, scoring="accuracy", n_jobs=-1)

    n_support = []
    for C in C_VALUES:
        pipe.set_params(svm__C=C).fit(X_train, y_train)
        n_support.append(int(pipe.named_steps["svm"].n_support_.sum()))

    df = pd.DataFrame({
        "C": C_VALUES,
        "train_acc": train_scores.mean(axis=1),
        "cv_acc": cv_scores.mean(axis=1),
        "gap": train_scores.mean(axis=1) - cv_scores.mean(axis=1),
        "support_vectors": n_support,
    })
    print("\n--- Effect of C (RBF kernel, gamma='scale') ---")
    print(df.round(4).to_string(index=False))

    best = df.loc[df["cv_acc"].idxmax()]
    print(f"\nBest CV accuracy {best['cv_acc']:.4f} at C = {best['C']:.3g}")
    print("Reading the table:")
    print(" - Very low C : wide margin, many support vectors, tends to UNDERFIT.")
    print(" - Very high C: narrow margin, few mistakes tolerated, train-CV gap grows (OVERFIT risk).")

    plot_c_sweep(C_VALUES, df["train_acc"], df["cv_acc"], n_support, len(X_train))
    return df


if __name__ == "__main__":
    run_c_sweep()
