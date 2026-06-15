# Baseline Models

## Purpose
Create simple, explainable baseline models to set reference performance: Logistic Regression and a Random Forest with sensible defaults.

## Implementation
- Baseline code: `src/train_rf.py` (Random Forest) and (optionally) `src/train_lr.py`.
- Train on preprocessed training data (`X_train`, `y_train`) without SMOTE for baseline, then with SMOTE as comparison.

## Notebook
- `notebooks/baseline_models.ipynb` should train baselines, record metrics, and save model artifacts to `models/` and reports to `reports/`.

## Expected outputs
- Baseline model pickles, evaluation metrics (F1, precision, recall), and simple plots (ROC, PR).
