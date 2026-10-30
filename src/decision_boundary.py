"""Day 2, Step 4: Visualise decision boundaries in 2D using PCA.

Run from the project root:  python -m src.decision_boundary

The 30-feature data cannot be drawn directly, so we scale, then project to 2
principal components. Boundaries here are for the 2D projection only, which is
an illustration, not the exact boundary of the 30D model.
"""
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.preprocess import prepare_data
from utils.config import RANDOM_STATE
from utils.plotting import plot_decision_boundaries


def run_decision_boundaries():
    X_train, _, _, _, y_train, _, _ = prepare_data()

    scaler = StandardScaler()
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X2 = pca.fit_transform(scaler.fit_transform(X_train))
    explained = pca.explained_variance_ratio_.sum()
    print(f"\n[pca] 2 components explain {explained:.1%} of the variance")

    # Figure 1: kernels side by side
    kernel_models = []
    for name, svc in [("Linear", SVC(kernel="linear", C=1)),
                      ("Polynomial (deg 3)", SVC(kernel="poly", degree=3, C=1)),
                      ("RBF", SVC(kernel="rbf", C=1, gamma="scale"))]:
        kernel_models.append((name, svc.fit(X2, y_train)))
    plot_decision_boundaries(kernel_models, X2, y_train,
                             "decision_boundary_kernels.png", "Kernels compared (PCA 2D)")

    # Figure 2: effect of C on the RBF kernel
    c_models = []
    for C in [0.01, 1, 100]:
        c_models.append((f"RBF, C={C}", SVC(kernel="rbf", C=C, gamma="scale").fit(X2, y_train)))
    plot_decision_boundaries(c_models, X2, y_train,
                             "decision_boundary_c.png", "Effect of C on RBF (PCA 2D)")
    return explained


if __name__ == "__main__":
    run_decision_boundaries()
