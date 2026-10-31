"""Reusable plotting helpers. Every function saves a PNG into data/figures/."""
import matplotlib

matplotlib.use("Agg")  # works without a display; remove if you want pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import (ConfusionMatrixDisplay, precision_recall_curve,
                             roc_auc_score, roc_curve, average_precision_score)

from utils.config import CLASS_NAMES, FIGURES_DIR, TARGET_COL

sns.set_theme(style="whitegrid")


def _save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved {path}")


def plot_class_distribution(df) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = df[TARGET_COL].map(CLASS_NAMES)
    sns.countplot(x=labels, ax=ax, hue=labels, palette="Set2", legend=False)
    ax.set_title("Class Distribution")
    ax.set_xlabel("Diagnosis")
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    _save(fig, "class_distribution.png")


def plot_feature_histograms(df, n_features: int = 10) -> None:
    cols = [c for c in df.columns if c != TARGET_COL][:n_features]
    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    labels = df[TARGET_COL].map(CLASS_NAMES)
    for ax, col in zip(axes.ravel(), cols):
        sns.histplot(data=df, x=col, hue=labels, kde=True, ax=ax, palette="Set2")
        ax.set_title(col, fontsize=10)
    fig.suptitle("Feature Distributions by Diagnosis (first 10 features)", fontsize=14)
    fig.tight_layout()
    _save(fig, "feature_histograms.png")


def plot_correlation_heatmap(df) -> None:
    fig, ax = plt.subplots(figsize=(16, 13))
    sns.heatmap(df.drop(columns=[TARGET_COL]).corr(), cmap="coolwarm", center=0,
                square=True, linewidths=0.3, cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    _save(fig, "correlation_heatmap.png")


def plot_scale_comparison(X_raw, X_scaled) -> None:
    """Boxplots before and after scaling, to show why SVM needs scaling."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    X_raw.iloc[:, :15].boxplot(ax=axes[0], rot=90)
    axes[0].set_title("Before Scaling (very different ranges)")
    X_scaled.iloc[:, :15].boxplot(ax=axes[1], rot=90)
    axes[1].set_title("After StandardScaler (comparable ranges)")
    fig.tight_layout()
    _save(fig, "scaling_comparison.png")


def plot_confusion_matrix(y_true, y_pred, name: str = "confusion_matrix.png",
                          title: str = "Confusion Matrix") -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=[CLASS_NAMES[0], CLASS_NAMES[1]],
        cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.grid(False)
    _save(fig, name)


# ----------------------------- Day 2 plots -----------------------------

def plot_kernel_comparison(df) -> None:
    """Bar chart of cross-validated accuracy per kernel (error bars = std)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(df["kernel"], df["cv_mean"], yerr=df["cv_std"], capsize=6,
           color=sns.color_palette("Set2", len(df)))
    ax.set_ylim(max(0, df["cv_mean"].min() - 0.05), 1.0)
    ax.set_ylabel("5-fold CV accuracy")
    ax.set_title("Kernel Comparison (default C and gamma)")
    for i, v in enumerate(df["cv_mean"]):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center")
    _save(fig, "kernel_comparison.png")


def plot_grid_heatmap(pivot, title: str, name: str) -> None:
    """Heatmap of CV accuracy for a C x gamma grid."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("gamma")
    ax.set_ylabel("C")
    _save(fig, name)


def plot_c_sweep(C_values, train_scores, cv_scores, n_support, n_train) -> None:
    """Train vs CV accuracy, and support-vector count, as C changes."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].semilogx(C_values, train_scores, "o-", label="Train accuracy")
    axes[0].semilogx(C_values, cv_scores, "s-", label="CV accuracy")
    axes[0].set_xlabel("C (log scale)")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Effect of C on RBF SVM")
    axes[0].legend()
    axes[1].semilogx(C_values, np.array(n_support) / n_train * 100, "o-", color="tab:red")
    axes[1].set_xlabel("C (log scale)")
    axes[1].set_ylabel("Support vectors (% of training data)")
    axes[1].set_title("Support Vectors vs C")
    fig.tight_layout()
    _save(fig, "c_sweep.png")


def plot_decision_boundaries(models, X2, y, name: str, suptitle: str) -> None:
    """Plot 2D decision boundaries. models = list of (title, fitted SVC)."""
    cmap = ListedColormap(["#e15759", "#4e79a7"])  # malignant, benign
    y = np.asarray(y)
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    axes = np.atleast_1d(axes)
    for ax, (title, model) in zip(axes, models):
        DecisionBoundaryDisplay.from_estimator(
            model, X2, response_method="predict", cmap=cmap, alpha=0.25,
            ax=ax, xlabel="PC1", ylabel="PC2")
        ax.scatter(X2[:, 0], X2[:, 1], c=y, cmap=cmap, edgecolors="white", s=25)
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                   s=90, facecolors="none", edgecolors="k", linewidths=0.8)
        ax.set_title(f"{title}\n{len(model.support_vectors_)} support vectors")
    fig.suptitle(suptitle + "  (red = malignant, blue = benign, circled = support vectors)",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, name)


# ----------------------------- Day 3 plots -----------------------------
# In all Day 3 plots, MALIGNANT is the positive class and `scores` are malignant scores.

def plot_roc_curves(curves, name: str = "roc_curve.png", marks=None,
                    title: str = "ROC Curve (malignant = positive)") -> None:
    """curves = [(label, y_true, scores)]; marks = [(fpr, tpr, label)] operating points."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for label, y, s in curves:
        pos = np.asarray(y) == 0
        fpr, tpr, _ = roc_curve(pos, s)
        ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {roc_auc_score(pos, s):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")
    mark_colors = ["tab:green", "tab:red", "tab:purple", "tab:orange"]
    for (fpr, tpr, label), color in zip(marks or [], mark_colors):
        ax.scatter([fpr], [tpr], s=90, zorder=5, label=label, color=color, edgecolors="k")
    ax.set_xlabel("False positive rate (benign flagged as malignant)")
    ax.set_ylabel("True positive rate (malignant caught)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    _save(fig, name)


def plot_pr_curves(curves, name: str = "pr_curve.png",
                   title: str = "Precision-Recall Curve (malignant = positive)") -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    prevalence = None
    for label, y, s in curves:
        pos = np.asarray(y) == 0
        prevalence = pos.mean()
        prec, rec, _ = precision_recall_curve(pos, s)
        ax.plot(rec, prec, lw=2, label=f"{label} (AP = {average_precision_score(pos, s):.3f})")
    if prevalence is not None:
        ax.axhline(prevalence, color="k", ls="--", alpha=0.4, label=f"Chance ({prevalence:.2f})")
    ax.set_xlabel("Recall (malignant caught)")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=9)
    _save(fig, name)


def plot_threshold_tradeoff(scores, y, t_default: float, t_chosen: float,
                            name: str = "threshold_tradeoff.png") -> None:
    """Recall and precision for malignant as the decision threshold moves."""
    scores, y = np.asarray(scores), np.asarray(y)
    ts = np.sort(np.unique(scores))
    recall = [(scores[y == 0] >= t).mean() for t in ts]
    precision = [(y[scores >= t] == 0).mean() if (scores >= t).any() else 1.0 for t in ts]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(ts, recall, lw=2, label="Recall (cancers caught)")
    ax.plot(ts, precision, lw=2, label="Precision")
    ax.axvline(t_default, color="grey", ls="--", label=f"Default threshold ({t_default:.2f})")
    ax.axvline(t_chosen, color="tab:red", ls="--", label=f"Chosen threshold ({t_chosen:.2f})")
    ax.set_xlabel("Malignant-score threshold (predict malignant if score >= threshold)")
    ax.set_ylabel("Rate")
    ax.set_title("Threshold Trade-off (training data, out-of-fold scores)")
    ax.legend(loc="lower left")
    _save(fig, name)


def plot_benchmark(df, name: str = "benchmark.png") -> None:
    """Grouped bars: nested-CV accuracy, malignant recall, ROC AUC per model."""
    metrics = [("accuracy", "Accuracy"), ("recall_malignant", "Malignant recall"),
               ("roc_auc", "ROC AUC")]
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / len(metrics)
    x = np.arange(len(df))
    for i, (col, label) in enumerate(metrics):
        ax.bar(x + i * width, df[f"{col}_mean"], width, yerr=df[f"{col}_std"],
               capsize=4, label=label)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(df["model"])
    ax.set_ylim(0.9, 1.01)
    ax.set_ylabel("Score (nested CV, mean +/- std)")
    ax.set_title("Model Benchmark")
    ax.legend(loc="lower right")
    _save(fig, name)


def plot_feature_importance(series, name: str = "feature_importance.png",
                            title: str = "Top features", n: int = 10) -> None:
    """Horizontal bars. Positive = pushes toward malignant."""
    top = series.reindex(series.abs().sort_values(ascending=False).index)[:n][::-1]
    colors = ["#e15759" if v > 0 else "#4e79a7" for v in top]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top.index, top.values, color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Weight (red = toward malignant, blue = toward benign)")
    ax.set_title(title)
    _save(fig, name)
