# Breast Cancer Detection with Support Vector Machines

A project that builds, tunes, and evaluates a Support Vector Machine (SVM) classifier to distinguish **malignant** from **benign** tumours using the Breast Cancer Wisconsin (Diagnostic) dataset.


## Project Roadmap

| Day | Focus | Status |
|-----|-------|--------|
| 1 | Data exploration, preprocessing, baseline linear SVM | Done |
| 2 | Kernel comparison, GridSearchCV tuning, C experiment, decision boundaries | Done |
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
├── data/                    # Dataset, results and figures (git-ignored)
├── src/                     # Pipeline steps
│   ├── eda.py               # Day 1: exploratory data analysis
│   ├── preprocess.py        # Day 1: train/test split and scaling
│   ├── baseline.py          # Day 1: baseline linear SVM
│   ├── kernels.py           # Day 2: kernel comparison with CV
│   ├── tuning.py            # Day 2: GridSearchCV tuning
│   ├── regularization.py    # Day 2: effect of C
│   ├── decision_boundary.py # Day 2: 2D decision boundaries (PCA)
│   └── main.py              # Day 2: runs the full pipeline
├── utils/                   # Reusable helpers
│   ├── config.py            # Paths and constants
│   ├── data_loader.py       # Dataset loading
│   ├── plotting.py          # Plot functions
│   └── metrics.py           # Evaluation helpers
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone and enter the project
```bash
git clone https://github.com/Muzammil592/Support_vector_machine_breast_cancer_detection
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

### 4. Run the pipelines
Run all commands from the project root.

```bash
python -m src.main

# Or run individual steps
python -m src.eda
python -m src.baseline
python -m src.kernels
python -m src.tuning
python -m src.regularization
python -m src.decision_boundary
```

Figures are saved to `data/figures/`, and tuning outputs to `data/`.

## Day 1 Results: Baseline

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Linear SVM (unscaled) | 0.956 | 0.947 | 0.986 | 0.966 |
| **Linear SVM (scaled, baseline)** | **0.974** | **0.986** | **0.972** | **0.979** |


Key takeaways:
- Classes are moderately imbalanced (62.7% benign), so stratified splitting is used.
- Many features are highly correlated (radius, perimeter, and area), which is expected from their geometry.
- Feature scaling improves the baseline and is essential for stable SVM training.
- Only about 7% of the training points are support vectors.

## Day 2 Results: Kernels and Tuning

All model selection below uses **5-fold cross-validation on the training set only**. The test set (114 samples) was used once, at the end.

### 1. Kernel comparison (default C=1, scaled)

| Kernel | CV accuracy | Std | Support vectors | Train accuracy |
|--------|-------------|-----|-----------------|----------------|
| Linear | 0.965 | 0.016 | 32 | 0.991 |
| Polynomial (deg 3) | 0.897 | 0.033 | 145 | 0.917 |
| **RBF** | **0.969** | 0.015 | 97 | 0.982 |
| Sigmoid | 0.967 | 0.014 | 66 | 0.956 |

Linear, RBF, and sigmoid are statistically indistinguishable (their differences are smaller than one standard deviation). The default polynomial kernel is clearly worse. This suggests the classes are close to linearly separable in the 30-dimensional space.

### 2. GridSearchCV (31 combinations x 5 folds = 155 fits)

| Rank | Parameters | CV accuracy | CV malignant recall |
|------|------------|-------------|---------------------|
| 1 (tie) | Linear, C=0.1 | 0.976 | 0.947 |
| 1 (tie) | RBF, C=10, gamma=0.01 | 0.976 | 0.953 |
| 3 | Linear, C=0.01 | 0.971 | 0.924 |
| 3 | RBF, C=1, gamma=0.01 | 0.971 | 0.935 |

**Selected model:** Linear kernel, C=0.1 (the simpler of the two tied models, ranked first by scikit-learn).

### 3. Baseline vs tuned model (held-out test set)

| Model | Accuracy | Errors (of 114) | Malignant recall |
|-------|----------|-----------------|------------------|
| Baseline (linear, C=1) | 0.974 | 3 | 0.976 (41/42) |
| **Tuned (linear, C=0.1)** | **0.982** | **2** | 0.976 (41/42) |

**Honest interpretation:** the tuned model gets one more test sample right. With 114 test samples, one sample is under one percentage point, which is within noise. Tuning gave a small, consistent CV gain (0.965 to 0.976) but no change in how many cancers are caught. The main lesson is that this dataset is easy for a well-scaled linear SVM, and tuning has limited headroom.

### 4. Effect of C (RBF kernel, gamma='scale')

| C | Train acc | CV acc | Train-CV gap | Support vectors |
|---|-----------|--------|--------------|-----------------|
| 0.001 | 0.626 | 0.626 | 0.000 | 340 |
| 0.01 | 0.626 | 0.626 | 0.000 | 341 |
| 0.1 | 0.956 | 0.943 | 0.013 | 199 |
| 0.316 | 0.981 | 0.969 | 0.012 | 138 |
| 1 | 0.985 | 0.969 | 0.016 | 97 |
| 10 | 0.996 | 0.969 | 0.026 | 84 |
| 100 | 1.000 | 0.967 | 0.033 | 64 |
| 1000 | 1.000 | 0.967 | 0.033 | 64 |

What this shows:
- **Very low C (0.01 or below):** the penalty is so weak that the model just predicts the majority class. Accuracy of 0.626 equals the benign share of the data. This is severe underfitting.
- **Moderate C (0.3 to 10):** the best and most stable region, with CV accuracy of about 0.969.
- **Very high C (100 or above):** training accuracy reaches 100%, the train-CV gap grows to 0.033, and CV accuracy does not improve. The model memorises the training set without generalising better.
- **Support vectors fall as C grows** (340 down to 64), because a narrower margin touches fewer points.

### 5. Decision boundaries (PCA, 2 components)

Two components explain 63.4% of the variance, so these plots are a simplified view of the real 30-dimensional problem. Even in 2D, the malignant and benign classes separate with a region of overlap in the middle. The polynomial kernel needs the most support vectors (133 in the 2D plot), and the RBF boundary bends more than the linear one. See `data/figures/decision_boundary_kernels.png` and `decision_boundary_c.png`.

### Takeaways
- Preprocessing inside a `Pipeline` keeps cross-validation honest.
- Kernel choice matters less than scaling and sensible C on this dataset.
- Always compare differences against the CV standard deviation and the test-set size before claiming an improvement.

## Generated Files

| File | Created by | Contents |
|------|-----------|----------|
| `data/breast_cancer.csv` | Day 1 | Dataset |
| `data/tuning_results.csv` | `tuning.py` | All grid combinations, ranked |
| `data/best_params.json` | `tuning.py` | Winning hyperparameters |
| `data/best_model.joblib` | `tuning.py` | Fitted tuned pipeline (for Day 3) |
| `data/figures/*.png` | all scripts | EDA, kernel, heatmap, C sweep, boundary, confusion matrix plots |

All of these are git-ignored and regenerated by running the scripts.

## Tech Stack

Python 3.9+, NumPy, pandas, scikit-learn, Matplotlib, Seaborn, joblib

## License

Released for educational purposes. The dataset is from the UCI Machine Learning Repository.
