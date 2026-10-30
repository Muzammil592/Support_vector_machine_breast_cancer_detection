"""Run the whole Day 2 pipeline: kernels -> tuning -> C experiment -> boundaries.

Usage (from the project root):  python -m src.run_day2
"""
from src.decision_boundary import run_decision_boundaries
from src.kernels import run_kernel_comparison
from src.regularization import run_c_sweep
from src.tuning import run_tuning


def banner(text: str) -> None:
    print("\n" + "=" * 60 + f"\n{text}\n" + "=" * 60)


def main():
    banner("DAY 2 | STEP 1: KERNEL COMPARISON")
    run_kernel_comparison()
    banner("DAY 2 | STEP 2: HYPERPARAMETER TUNING (GridSearchCV)")
    run_tuning()
    banner("DAY 2 | STEP 3: EFFECT OF C")
    run_c_sweep()
    banner("DAY 2 | STEP 4: DECISION BOUNDARIES (PCA 2D)")
    run_decision_boundaries()
    print("\nDay 2 complete. Figures are in data/figures/")


if __name__ == "__main__":
    main()
