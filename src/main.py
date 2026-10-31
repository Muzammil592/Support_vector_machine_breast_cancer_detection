"""Run the whole Day 3 pipeline: evaluation -> benchmark -> interpretation.

Usage (from the project root):  python -m src.run_day3
Needs the Day 2 model; if it is missing, Day 2 tuning is run automatically.
"""
from src.benchmark import run_benchmark
from src.evaluate import run_evaluation
from src.interpret import run_interpretation


def banner(text: str) -> None:
    print("\n" + "=" * 60 + f"\n{text}\n" + "=" * 60)


def main():
    banner("DAY 3 | STEP 1: FINAL EVALUATION (malignant = positive)")
    run_evaluation()
    banner("DAY 3 | STEP 2: BENCHMARK (SVM vs LogReg vs Random Forest)")
    run_benchmark()
    banner("DAY 3 | STEP 3: INTERPRETATION AND ERROR ANALYSIS")
    run_interpretation()
    print("\nDay 3 complete. Figures are in data/figures/")


if __name__ == "__main__":
    main()
