"""Central configuration for the project."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FIGURES_DIR = DATA_DIR / "figures"
DATASET_PATH = DATA_DIR / "breast_cancer.csv"

# Day 2 outputs (all inside data/, which is git-ignored)
TUNING_RESULTS_PATH = DATA_DIR / "tuning_results.csv"
BEST_PARAMS_PATH = DATA_DIR / "best_params.json"
BEST_MODEL_PATH = DATA_DIR / "best_model.joblib"

# Day 3 outputs
FINAL_RESULTS_PATH = DATA_DIR / "final_results.json"
BENCHMARK_RESULTS_PATH = DATA_DIR / "benchmark_results.csv"
FEATURE_IMPORTANCE_PATH = DATA_DIR / "feature_importance.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
TARGET_COL = "target"

# Day 3 settings
TARGET_RECALL = 0.98      # desired share of malignant cases caught (threshold tuning)
N_BOOTSTRAP = 2000        # bootstrap resamples for confidence intervals
NESTED_REPEATS = 2        # repeats of the outer CV in the benchmark

# In scikit-learn's dataset: 0 = malignant, 1 = benign
CLASS_NAMES = {0: "malignant", 1: "benign"}
