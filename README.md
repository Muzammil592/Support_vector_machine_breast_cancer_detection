# Breast Cancer Detection with Support Vector Machines

A project that builds, tunes, evaluates, and benchmarks a Support Vector Machine (SVM) classifier to distinguish **malignant** from **benign** tumours using the Breast Cancer Wisconsin (Diagnostic) dataset.

> **Educational project only.** This is not a medical device and must not be used for diagnosis.

## Headline Results

| Metric (test set, malignant = positive) | Value | 95% CI |
|------------------------------------------|-------|--------|
| Accuracy | 98.2% | [95.6%, 100%] |
| Recall (cancers caught) | 97.6% (41 of 42) | [92.1%, 100%] |
| Precision | 97.6% | [92.1%, 100%] |
| ROC AUC | 0.994 | [0.979, 1.000] |

Final model: linear-kernel SVM (C=0.1) with standardised features. In nested cross-validation on all 569 samples it reaches 97.7% accuracy and 95.8% malignant recall, ahead of Logistic Regression and Random Forest, though the gap to Logistic Regression is within noise.

## Project Roadmap

| Day | Focus | Status |
|-----|-------|--------|
| 1 | Data exploration, preprocessing, baseline linear SVM | Done |
| 2 | Kernel comparison, GridSearchCV tuning, C experiment, decision boundaries | Done |
| 3 | Final evaluation, threshold tuning, model benchmark, interpretation | Done |

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
│   ├── evaluate.py          # Day 3: final evaluation, CIs, threshold tuning
│   ├── benchmark.py         # Day 3: SVM vs Logistic Regression vs Random Forest
│   ├── interpret.py         # Day 3: feature weights and error analysis
│   ├── predict.py           # Day 3: predict with the saved model
│   └── main.py              # Day 3: runs the full pipeline
├── utils/                   # Reusable helpers
│   ├── config.py            # Paths and constants
│   ├── data_loader.py       # Dataset loading
│   ├── plotting.py          # Plot functions
│   └── metrics.py           # Metrics, thresholds, bootstrap CIs
├── requirements.txt
└── README.md
```

## Getting Started

```bash
git clone https://github.com/Muzammil592/Support_vector_machine_breast_cancer_detection
cd svm-breast-cancer
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Run all commands from the project root.

```bash
python -m src.main

# Predict with the saved model
python -m src.predict --index 0 5 12
python -m src.predict --csv my_samples.csv
```

Figures are saved to `data/figures/`, and tables and the saved model to `data/`.

## Day 1 Results: Baseline

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Linear SVM (unscaled) | 0.956 | 0.947 | 0.986 | 0.966 |
| **Linear SVM (scaled, baseline)** | **0.974** | **0.986** | **0.972** | **0.979** |

*metrics treat benign (class 1) as positive. From Day 3 onwards, malignant is the positive class.*

- Classes are moderately imbalanced (62.7% benign), so stratified splitting is used.
- Many features are highly correlated (radius, perimeter, and area).
- Feature scaling improves the baseline and is essential for stable SVM training.
- Only about 7% of training points are support vectors.

## Initial Results: Kernels and Tuning

All selection uses **5-fold cross-validation on the training set only**.

| Kernel (C=1) | CV accuracy | Support vectors |
|--------------|-------------|-----------------|
| Linear | 0.965 | 32 |
| Polynomial | 0.897 | 145 |
| RBF | 0.969 | 97 |
| Sigmoid | 0.967 | 66 |

Linear, RBF, and sigmoid are within one standard deviation of each other. GridSearchCV (31 combinations, 155 fits) selected **linear, C=0.1** (CV accuracy 0.976), tied with RBF at C=10, gamma=0.01. On the test set the tuned model made 2 errors against 3 for the baseline, which is within noise.

**Effect of C (RBF):**
- At C of 0.01 or below, the model collapses to predicting the majority class (accuracy 0.626).
- C between 0.3 and 10 is the stable sweet spot (CV about 0.969).
- At C of 100 or more, training accuracy reaches 100% and the train-CV gap grows to 0.033 with no CV gain.
- The support-vector count falls from 340 to 64 as C grows.

## Final Results: Evaluation, Benchmark, Interpretation

### 1. Final test-set evaluation

| | Predicted malignant | Predicted benign |
|---|---|---|
| **Actually malignant (42)** | 41 | 1 |
| **Actually benign (72)** | 1 | 71 |

Precision, recall, and F1 for malignant are all 0.976. ROC AUC is 0.994 and average precision is 0.992. Because the test set has only 114 samples, the 95% bootstrap intervals are wide: the true recall could plausibly be anywhere from about 92% to 100%.

### 2. Threshold tuning (a negative result)

I lowered the decision threshold to reach 98% malignant recall in training cross-validation, then applied it to the test set.

| Setting | Recall | Precision | Cancers missed | False alarms |
|---------|--------|-----------|----------------|--------------|
| Train CV, default threshold | 0.947 | 0.988 | 9 | 2 |
| Train CV, chosen threshold (-0.66) | 0.982 | 0.918 | 3 | 15 |
| **Test, default threshold** | 0.976 | 0.976 | 1 | 1 |
| **Test, chosen threshold** | 0.976 | 0.872 | 1 | 6 |

On the training data, the lower threshold caught 6 more cancers at the cost of 13 more false alarms. On the test set it caught **no additional cancer** and added 5 false alarms. The single missed cancer has a score of -1.21, deep on the benign side, so no reasonable threshold would catch it without flooding the results with false alarms. The default threshold is the better choice for this model.

### 3. Benchmark: SVM vs Logistic Regression vs Random Forest

**A) Hold-out test (tuned on train, scored once on test)**

| Model | Best parameters | Test accuracy | Malignant recall | ROC AUC |
|-------|-----------------|---------------|------------------|---------|
| **SVM** | linear, C=0.1 | **0.983** | **0.976** | 0.994 |
| Logistic Regression | C=0.1 | 0.974 | 0.952 | 0.996 |
| Random Forest | 200 trees, defaults | 0.956 | 0.929 | 0.993 |

**B) Nested cross-validation on all 569 samples (5 folds x 2 repeats, mean +/- std)**

| Model | Accuracy | Malignant recall | ROC AUC |
|-------|----------|------------------|---------|
| **SVM** | **0.977 +/- 0.010** | **0.958 +/- 0.026** | 0.995 +/- 0.005 |
| Logistic Regression | 0.974 +/- 0.014 | 0.944 +/- 0.038 | 0.995 +/- 0.005 |
| Random Forest | 0.955 +/- 0.013 | 0.929 +/- 0.039 | 0.990 +/- 0.008 |

**Interpretation:** SVM and Logistic Regression are statistically indistinguishable (a 0.4 point accuracy gap against a standard deviation of about 1 point), which is expected since both are linear models here. Random Forest is about 2 points lower on accuracy, a gap larger than its standard deviation, so it is plausibly worse on this dataset. Nested CV is the more reliable comparison because all 569 samples are used for scoring and tuning never sees the scoring fold.

### 4. What the model relies on

The strongest weights in the linear SVM all point toward malignant: worst texture (0.45), radius error (0.39), worst symmetry (0.38), area error (0.35), and worst radius (0.34). Because radius, perimeter, and area are highly correlated, their weight is shared, so individual weights should not be read as single causes.

### 5. Error analysis

Two test samples were misclassified:
- **False alarm (sample 541):** a benign case with a malignant score of 0.37, close to the boundary and a reasonable mistake.
- **Missed cancer (sample 73):** a malignant case with a score of -1.21, a confident mistake far from the boundary. This is the case worth reviewing in a real setting.

## Limitations

- **Small dataset and test set:** 114 test samples mean one case moves recall by 2.4 points. All confidence intervals reflect this.
- **One dataset, one split:** results are from a single stratified split plus nested CV, not from independent external data.
- **Clean, curated features:** the features are pre-computed from images. Real-world performance would depend on image quality and acquisition.
- **Not clinically validated:** no clinical trial, calibration to real prevalence, or regulatory review.
- **Model scores are not probabilities:** SVM scores indicate distance from the boundary, not calibrated risk.

## Possible Extensions

- Calibrate scores with `CalibratedClassifierCV`
- Add Gradient Boosting and k-NN to the benchmark
- Build a Streamlit app on top of `src/predict.py`
- Repeat on a different dataset, such as text classification with TF-IDF

## Generated Files

| File | Created by | Contents |
|------|-----------|----------|
| `data/breast_cancer.csv` | Day 1 | Dataset |
| `data/tuning_results.csv`, `best_params.json`, `best_model.joblib` | `tuning.py` | Grid results and the saved model |
| `data/final_results.json` | `evaluate.py` | Final metrics, CIs, threshold results |
| `data/benchmark_results.csv` | `benchmark.py` | Model comparison table |
| `data/feature_importance.csv` | `interpret.py` | Linear SVM weights |
| `data/figures/*.png` | all scripts | All plots |

All of these are git-ignored and regenerated by running the scripts.

## Tech Stack

Python 3.9+, NumPy, pandas, scikit-learn, Matplotlib, Seaborn, joblib

## License

Released for educational purposes. The dataset is from the UCI Machine Learning Repository.
