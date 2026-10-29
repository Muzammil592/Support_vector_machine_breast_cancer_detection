"""Step 3: Baseline linear SVM, with and without scaling.

Run from the project root:  python -m src.baseline
"""
from sklearn.svm import SVC

from src.preprocess import prepare_data
from utils.config import RANDOM_STATE
from utils.metrics import evaluate
from utils.plotting import plot_confusion_matrix


def run_baseline():
    X_train, X_test, X_train_s, X_test_s, y_train, y_test, _ = prepare_data(make_plot=True)

    # 1) Linear SVM on UNSCALED data (to see why scaling matters)
    raw_model = SVC(kernel="linear", random_state=RANDOM_STATE, max_iter=1_000_000)
    raw_model.fit(X_train, y_train)
    raw_results = evaluate(y_test, raw_model.predict(X_test), "Linear SVM (unscaled)")

    # 2) Linear SVM on SCALED data (the real baseline)
    model = SVC(kernel="linear", C=1.0, random_state=RANDOM_STATE)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    results = evaluate(y_test, y_pred, "Linear SVM (scaled) - BASELINE")
    plot_confusion_matrix(y_test, y_pred, "baseline_confusion_matrix.png",
                          "Baseline Linear SVM")

    print("\n--- Support vectors ---")
    print(f"Total support vectors: {model.n_support_.sum()} "
          f"(per class: {{'malignant': int(model.n_support_[0]), 'benign': int(model.n_support_[1])}})")
    print(f"That is {model.n_support_.sum() / len(X_train_s):.1%} of the training data.")

    print("\n--- Unscaled vs Scaled accuracy ---")
    print(f"Unscaled: {raw_results['accuracy']:.4f} | Scaled: {results['accuracy']:.4f}")
    return results


if __name__ == "__main__":
    run_baseline()
