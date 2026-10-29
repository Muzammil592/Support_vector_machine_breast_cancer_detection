"""Run the whole Day 1 pipeline: EDA -> preprocessing -> baseline.

Usage (from the project root):  python -m src.run_day1
"""
from src.baseline import run_baseline
from src.eda import run_eda


def main():
    print("=" * 60 + "\nSTEP 1: EXPLORATORY DATA ANALYSIS\n" + "=" * 60)
    run_eda()
    print("\n" + "=" * 60 + "\n STEPS 2-3: PREPROCESSING + BASELINE SVM\n" + "=" * 60)
    run_baseline()
    print("\n Figures are in data/figures/")


if __name__ == "__main__":
    main()
