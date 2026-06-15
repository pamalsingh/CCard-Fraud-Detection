# Data Access and Preprocessing

## Purpose
Document data loading, validation, cleaning, and preprocessing steps used in the project.

## Contents
- Data source: `data/raw/creditcard.csv` and expected schema.
- Loading helper: `src/data_loader.py` — how to call it.
- Preprocessing steps (dedupe, train/test split, scaling): implemented in `src/preprocessing.py`.
- Optional resampling (SMOTE) used in `src/imbalance_handler.py`.

## How to run
1. Install requirements: `pip install -r requirements.txt`.
2. From repo root run preprocessing script or call pipeline:
```bash
python pipelines/train_pipeline.py --tuning-file reports/tuning_results.json
```

## Expected outputs
- `X_train`, `X_test`, `y_train`, `y_test` produced by `preprocess_data`.
- Scalers saved to `models/amount_scaler.pkl` and `models/feature_scaler.pkl` by the pipeline.
