# Breast Cancer Detection with Support Vector Machines

A 3-day project that builds, tunes, and evaluates a Support Vector Machine (SVM) classifier to distinguish **malignant** from **benign** tumours using the Breast Cancer Wisconsin (Diagnostic) dataset.

> **Status:** complete (EDA, preprocessing, baseline model). (kernels and tuning) and (evaluation and write-up) to follow.

## Project Roadmap

| Day | Focus | Status |
|-----|-------|--------|
| 1 | Data exploration, preprocessing, baseline linear SVM | Done |
| 2 | Kernel comparison, hyperparameter tuning (GridSearchCV), decision boundaries | Planned |
| 3 | Full evaluation (ROC, recall focus), model benchmarking, final report | Planned |

## Dataset

- **Source:** Breast Cancer Wisconsin (Diagnostic), bundled with scikit-learn (originally from the UCI Machine Learning Repository)
- **Samples:** 569 (357 benign, 212 malignant)
- **Features:** 30 numeric features computed from digitised images of a fine needle aspirate of breast tissue (mean, standard error, and worst value of radius, texture, perimeter, area, smoothness, and more)
- **Target:** `0` = malignant, `1` = benign
- **Missing values:** none

The dataset is generated automatically into `data/breast_cancer.csv` on first run. The `data/` folder is git-ignored.

## Project Structure

```
svm-breast-cancer/
├── data/                  # Dataset and generated figures (git-ignored)
├── src/                   # Pipeline steps
│   ├── eda.py             # Exploratory data analysis
│   ├── preprocess.py      # Train/test split and scaling
│   ├── baseline.py        # Baseline linear SVM
│   └── main.py            
├── utils/                 # Reusable helpers
│   ├── config.py          # Paths and constants
│   ├── data_loader.py     # Dataset loading
│   ├── plotting.py        # Plot functions
│   └── metrics.py         # Evaluation helpers
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone and enter the project
```bash
git clone <your-repo-url>
cd svm-breast-cancer
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Code
Run all commands from the project root.

```bash
python -m src.run_day1     # full pipeline
python -m src.eda          # EDA only
python -m src.baseline     # preprocessing + baseline only
```

Figures are saved to `data/figures/`.

## Day 1 Results

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Linear SVM (unscaled) | 0.956 | 0.947 | 0.986 | 0.966 |
| **Linear SVM (scaled, baseline)** | **0.974** | **0.986** | **0.972** | **0.979** |

Key takeaways:
- Classes are moderately imbalanced (62.7% benign), so stratified splitting is used.
- Many features are highly correlated (radius, perimeter, and area), which is expected from their geometry.
- Feature scaling improves the baseline and is essential for stable SVM training.
- Only about 7% of the training points are support vectors, so the decision boundary depends on a small subset of the data.

## Tech Stack

Python 3.9+, NumPy, pandas, scikit-learn, Matplotlib, Seaborn

## License

Released for educational purposes. The dataset is from the UCI Machine Learning Repository.
