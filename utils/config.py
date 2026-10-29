"""Central configuration for the project."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FIGURES_DIR = DATA_DIR / "figures"
DATASET_PATH = DATA_DIR / "breast_cancer.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "target"

# In scikit-learn's dataset: 0 = malignant, 1 = benign
CLASS_NAMES = {0: "malignant", 1: "benign"}
