"""Day 3, Step 3: Interpret the model and study its mistakes.

Run from the project root:  python -m src.interpret
"""
import pandas as pd
from sklearn.inspection import permutation_importance

from src.evaluate import load_best_model
from src.preprocess import prepare_data
from utils.config import CLASS_NAMES, FEATURE_IMPORTANCE_PATH, RANDOM_STATE
from utils.metrics import malignant_scores
from utils.plotting import plot_feature_importance


def run_interpretation():
    X_train, X_test, _, _, y_train, y_test, _ = prepare_data()
    model = load_best_model(X_train, y_train)
    svm = model.named_steps["svm"]

    # ---------- Feature importance ----------
    if svm.kernel == "linear":
        # Features are standardised, so weights are comparable.
        # coef_ > 0 pushes toward class 1 (benign), so flip the sign for "toward malignant".
        importance = pd.Series(-svm.coef_[0], index=X_train.columns)
        title = "Linear SVM weights (standardised features)"
        print("\n--- Linear SVM weights: top 10 features by |weight| ---")
    else:
        result = permutation_importance(model, X_test, y_test, n_repeats=20,
                                        random_state=RANDOM_STATE, scoring="accuracy")
        importance = pd.Series(result.importances_mean, index=X_train.columns)
        title = "Permutation importance (accuracy drop)"
        print("\n--- Permutation importance: top 10 features ---")

    top = importance.reindex(importance.abs().sort_values(ascending=False).index)
    print(top.head(10).round(3).to_string())
    top.to_csv(FEATURE_IMPORTANCE_PATH, header=["weight"])
    plot_feature_importance(importance, title=title)
    print("\nNote: many features are strongly correlated (radius, perimeter, area), so the "
          "weight is shared between them. Do not read a single weight as 'the' cause.")

    # ---------- Misclassified test samples ----------
    scores = malignant_scores(model, X_test)
    pred = model.predict(X_test)
    report = pd.DataFrame({
        "true": y_test.map(CLASS_NAMES), "predicted": pd.Series(pred, index=y_test.index).map(CLASS_NAMES),
        "malignant_score": scores}, index=y_test.index)
    errors = report[report["true"] != report["predicted"]]
    print(f"\n--- Misclassified test samples: {len(errors)} of {len(report)} ---")
    print(errors.round(3).to_string() if len(errors) else "None")
    for _, row in errors.iterrows():
        kind = "MISSED CANCER (false negative)" if row["true"] == "malignant" else "False alarm (false positive)"
        print(f" - {kind}")

    print("\n--- 5 most borderline test samples (score closest to the decision boundary) ---")
    borderline = report.loc[report["malignant_score"].abs().sort_values().index[:5]]
    print(borderline.round(3).to_string())
    return importance, errors


if __name__ == "__main__":
    run_interpretation()
