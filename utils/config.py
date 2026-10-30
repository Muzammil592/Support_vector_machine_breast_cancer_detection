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

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
TARGET_COL = "target"

# In scikit-learn's dataset: 0 = malignant, 1 = benign
CLASS_NAMES = {0: "malignant", 1: "benign"}
