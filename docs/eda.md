# Exploratory Data Analysis (EDA)

## Purpose
Summarize dataset characteristics, distributions, imbalance, missing values, and key observations that inform modeling decisions.

## Suggested contents
- Class balance and event rate.
- Feature distributions (histograms, boxplots) and `Amount` scaling.
- Correlation matrix and highly correlated features.
- Time-based checks (if `Time` exists) and duplicate rows.
- Data-quality checks: missing values, outliers, anomalous rows.

## Notebooks and scripts
- Recommended notebook: `notebooks/eda.ipynb` that imports `src/preprocessing.py` outputs and generates plots saved to `reports/`.

## Key artifacts to save
- `reports/eda_summary.txt` or `reports/eda_plots.zip` containing principal plots and a short bullets list of insights.
