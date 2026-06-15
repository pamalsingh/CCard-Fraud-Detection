# Problem Statement and Scope

## Problem Statement

Credit card fraud causes financial losses and erodes customer trust. The goal of this project is to build a reproducible, well-documented machine learning solution to detect fraudulent transactions in the provided credit card dataset, focusing on maximizing detection performance while controlling false positives and producing interpretable evaluation and deployment artifacts.

## Objectives

- Prepare and document a reproducible data pipeline (ingest, clean, preprocess).
- Perform exploratory data analysis and identify data quality issues and class imbalance characteristics.
- Implement baseline and advanced models (Random Forest, XGBoost, DNN), including hyperparameter tuning and class-imbalance handling.
- Produce evaluation reports with metrics, visualizations, and error-analysis samples.
- Deliver a minimal deployment prototype (Streamlit or FastAPI) demonstrating model inference.
- Provide a final presentation and a reproducible project repository with instructions.

## Scope (In-Scope)

- Use the provided `data/raw/creditcard.csv` dataset as the single data source.
- Data preprocessing: deduplication, scaling, train/test split, optional SMOTE for imbalance.
- Models: Random Forest, XGBoost, and a feed-forward DNN.
- Hyperparameter tuning via `GridSearchCV` or `RandomizedSearchCV` (RF/XGB) and manual grid search for DNN.
- Evaluation: ROC, precision-recall, F1, confusion matrix, and sample error inspection.
- Save trained models, scalers, and evaluation artifacts to `models/` and `reports/`.

## Out of Scope

- External data collection or feature enrichment beyond the supplied dataset.
- Production-grade deployment (authentication, autoscaling, full CI/CD).
- Extensive model explainability beyond basic feature importance and error examples.

## Deliverables

- Problem statement and scope (this document).
- Data access & preprocessing notebook or script (`notebooks/` or `src/`).
- EDA notebook with observations and data-quality checks.
- Baseline model notebook and advanced model notebook with training code and saved artifacts.
- Model evaluation report with metrics, plots, and sample errors saved under `reports/`.
- Deployment prototype (Streamlit or FastAPI) under `dashboard/` or `api/`.
- Final presentation deck (PDF or slides) and a comprehensive `README.md` with reproduction steps.

## Success Criteria

- Reproducibility: a user can run the pipeline and reproduce models and reports using documented commands.
- Performance: improved F1 and precision-recall tradeoffs over baseline (quantitative thresholds to be defined during model evaluation).
- Artifacts: trained model files, scalers, and evaluation plots saved and viewable in `reports/` and `models/`.

## Constraints and Assumptions

- Work is limited to the compute resources available locally; large-grid searches may be subsampled or switched to randomized search.
- TensorFlow/XGBoost may be optional based on the environment; the code will check for availability and fall back gracefully.
- Dataset schema is assumed stable as provided in `data/raw/creditcard.csv`.

## Risks and Mitigations

- Long tuning times: mitigate with subsampling (`--n-samples`) or `RandomizedSearchCV`.
- Imbalanced labels causing misleading accuracy: use F1/AUPRC and resampling (SMOTE) and report false-negative cases.

## Timeline & Next Steps

1. Produce `Data access and preprocessing` notebook/script (this week).
2. Create `EDA` notebook and record key observations.
3. Implement baseline models and evaluation notebook.
4. Implement advanced models, tuning, and create evaluation reports.
5. Build a minimal deployment prototype and final presentation.

## Contact / Ownership

Primary developer: repository owner (you). For questions or changes, create an issue in the project root.
