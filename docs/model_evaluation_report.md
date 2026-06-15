# Model Evaluation Report

## Purpose
Describe evaluation methodology, metrics, plots, and provide sample error cases for manual inspection.

## Metrics
- Confusion matrix, precision, recall, F1-score, ROC AUC, PR AUC.
- Per-class and overall metrics; emphasize recall on the positive (fraud) class.

## Plots
- ROC curve, Precision-Recall curve, Confusion matrix heatmap.
- Calibration plot and feature importance for tree models.

## Error analysis
- Save sample false negatives and false positives to CSV (already present under `reports/`).

## Expected artifacts
- `reports/*_classification_report.txt`, `reports/*_sample_predictions.csv`, `reports/*_roc.png`, `reports/*_pr.png`.
