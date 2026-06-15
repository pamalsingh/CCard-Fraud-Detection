# Credit Card Fraud Detection - Streamlit App

This repository contains a Streamlit dashboard to load pre-trained models and run fraud predictions on uploaded transaction data.

How to use

1. Place your trained model and optional preprocessing artifacts in the `models/` folder. Supported files:
   - `xgboost_pre_smote.pkl`, `random_forest_pre_smote.pkl`, etc.
   - `preprocessor.pkl`, `scaler.pkl`, `encoder.pkl`, `feature_columns.pkl`

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

4. In the sidebar select the model file, then upload a CSV or Excel file containing transaction records. The app will preview the data, allow you to select feature columns if needed, apply preprocessing (if available), and run predictions.

API

There is also a minimal FastAPI server at `api/main.py` with an endpoint `/predict_csv` which accepts a file upload and a `model_file` query parameter.

Notes

- The app does not retrain models; it only loads saved models for inference.
- If your DNN models require TensorFlow/Keras, ensure `tensorflow` is installed.
- The app attempts to infer feature columns from `feature_columns.pkl` or from model attributes. If inference fails, select features manually in the UI.
