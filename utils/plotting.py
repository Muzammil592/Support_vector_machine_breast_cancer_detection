"""Reusable plotting helpers. Every function saves a PNG into data/figures/."""
import matplotlib

matplotlib.use("Agg")  # works without a display; remove if you want pop-up windows
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

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
