# Reproducibility & Project README

## Purpose
Provide instructions so a user can reproduce data preprocessing, model training, evaluation, and run the demo locally.

## Minimal reproduction steps
1. Clone repository and `cd` into project root.
2. Create Python environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```
3. Run the training pipeline (optionally use tuning results):
```bash
python pipelines/train_pipeline.py --tuning-file reports/tuning_results.json
```
4. Run the demo (Streamlit):
```bash
streamlit run dashboard/app.py
```

## Files to check after running
- `models/` for model artifacts
- `reports/` for evaluation artifacts and plots
