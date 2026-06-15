# Advanced Models

## Purpose
Document advanced modeling efforts: hyperparameter tuning, ensembling, and DNN training.

## Models
- Random Forest (tuned with `GridSearchCV` or `RandomizedSearchCV`) — `src/tune_models.py`.
- XGBoost (tuned) — `src/train_xgb.py` & `src/tune_models.py`.
- DNN (feed-forward) with early stopping; manual grid-search implemented in `src/tune_models.py` which calls `src/train_dnn.py`.

## Notebooks
- `notebooks/advanced_models.ipynb` should include tuning runs, validation curves, and final model selection.

## Artifacts
- Save tuned parameter JSON to `reports/tuning_results.json` and final model artifacts to `models/`.
