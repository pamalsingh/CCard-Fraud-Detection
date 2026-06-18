# 🛡️ Credit Card Fraud Detection

## 📌 Project Overview
This project aims to detect fraudulent credit card transactions using machine learning techniques.  
The primary business goal is to **minimize financial losses due to fraud** while maintaining a **smooth customer experience** by reducing false alarms.

---

## 🎯 Business Goal
- Protect banks and customers from fraudulent transactions.
- Balance **Recall (catching fraud)** with **Precision (avoiding false alarms)**.
- Ensure trust and customer satisfaction by minimizing false positives.

---

## 📂 Dataset
- Source: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Records: **284,807 transactions**
- Fraud cases: **492 (0.172%)**
- Features:
   - `Time`: Seconds elapsed since first transaction (not clock time).
   - `Amount`: Transaction amount (scaled using RobustScaler).
   - `V1–V28`: PCA-transformed features (anonymized).
   - `Class`: Target variable (0 = genuine, 1 = fraud).

---

## 🔍 Exploratory Data Analysis (EDA)
- **Data Types:** Mostly numeric, no categorical features except target.
- **Missing Values:** None.
- **Duplicates:** None.
- **Class Distribution:** Highly imbalanced (fraud <0.2%).
- **Distribution Analysis:** `Amount` skewed, `Time` cyclical.
- **Outliers:** Large transaction amounts → handled with RobustScaler.
- **Correlation:** PCA features uncorrelated by design.
- **Visualizations:** Histograms, boxplots, heatmaps, class imbalance plots.

### 🧩 EDA Findings → Decisions
- Skewed `Amount` → applied **RobustScaler**.  
- Severe class imbalance → applied **SMOTE** on training data.  
- Weak correlations → relied on ML models for non-linear learning.  
- No missing/duplicates → skipped imputation/cleaning.

---

## ⚙️ Modeling
Models tested:
- **Random Forest (RF)** – baseline ensemble model.
- **XGBoost** – selected final model for best balance of precision & recall.
- **Deep Neural Network (DNN)** – experimented with BatchNorm & Dropout.

### 📊 Evaluation Metrics
- Precision, Recall, F1-score
- ROC-AUC, PR-AUC
- Latency (prediction speed)

---

## 📈 Results
- **XGBoost selected** for deployment:
   - Best trade-off between catching fraud and minimizing false alarms.
   - Robust to class imbalance.
   - Boosting approach captured minority fraud patterns effectively.

---

## 🚀 Deployment
- Built a **Streamlit dashboard**:
   - Accepts CSV/Excel input.
   - Returns fraud probability & prediction.
- Deployed on **AWS ECS** with Docker & ECR.
- Configured networking & security groups for public access.

---

## 🧮 Confusion Matrix (Business Terms)
- **True Positive (TP):** Fraud caught → prevents financial loss.
- **True Negative (TN):** Genuine transaction allowed → smooth experience.
- **False Positive (FP):** Genuine blocked → customer frustration.
- **False Negative (FN):** Fraud missed → direct financial loss.

---

## ⚙️ Usage Instructions

### 🔧 Model Tuning
To tune models and get the best combination of hyperparameters:
```bash
python -m src.tune_models --model all --out reports/tuning_results.json
```

### 🏋️ Training

To retrain models (should only be used by Data Scientists under supervision if having access):

```bash
python pipelines/train_pipeline.py --tuning-file reports/tuning_results.json
```

### 💻 How to Use

1. Place your trained model and optional preprocessing artifacts in the `models/` folder. Supported files:

- `xgboost_pre_smote.pkl`, `random_forest_pre_smote.pkl`, etc.
- `preprocessor.pkl`, `scaler.pkl`, `feature_columns.pkl`, `amount_scaler.pkl`, `feature_scaler.pkl`
2. Install dependencies:

```bash
pip install -r requirements.txt
```
3. Run the Streamlit app:

```bash
streamlit run app.py
```
4. In the sidebar select the model file, then upload a CSV or Excel file containing transaction records.

The app will preview the data, allow you to select feature columns if needed, apply preprocessing (if available), and run predictions.

---

## 🌐 API

A minimal **FastAPI server** is available at `api/main.py` with an endpoint:

- **`/predict_csv`** → accepts a file upload and a `model_file` query parameter.

---

## 📝 Notes

- The app does **not retrain models**; it only loads saved models for inference.
- If your DNN models require TensorFlow/Keras, ensure `tensorflow` is installed.
- The app attempts to infer feature columns from `feature_columns.pkl` or from model attributes.
- If inference fails, select features manually in the UI.

---

## ⚠️ Risks & Future Work

- **Risks:** Class imbalance, model drift, explainability challenges.
- **Future Enhancements:**

- SHAP (SHapley Additive exPlanations) helps explain individual predictions for interpretability
- Threshold tuning for better trade-offs.
- Drift detection & MLOps pipeline.
- Cloud scaling for production workloads.

---

## 👨‍💻 Author

**Pamal Harpreet Singh**  

Senior Lead Test Engineer | Expanding into AI/ML/DL for fintech solutions

📍 Chandigarh, India

---