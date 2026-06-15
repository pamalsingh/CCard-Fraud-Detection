import os
import io
import pickle
from typing import Optional

import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px


MODELS_DIR = "models"


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_model(path: str):
    try:
        return joblib.load(path)
    except Exception:
        try:
            return load_pickle(path)
        except Exception as e:
            st.error(f"Failed to load model {path}: {e}")
            return None


def get_available_models():
    if not os.path.isdir(MODELS_DIR):
        return []
    files = [f for f in os.listdir(MODELS_DIR) if os.path.isfile(os.path.join(MODELS_DIR, f))]
    model_files = [f for f in files if f.lower().endswith(('.pkl', '.joblib'))]
    return model_files


def apply_preprocessors(df: pd.DataFrame, feature_cols, models_folder=MODELS_DIR):
    X = df.copy()
    # Select features
    Xf = X.loc[:, feature_cols].copy()

    # Apply preprocessor if exists
    preproc_path = os.path.join(models_folder, 'preprocessor.pkl')
    scaler_path = os.path.join(models_folder, 'scaler.pkl')
    encoder_path = os.path.join(models_folder, 'encoder.pkl')

    try:
        if os.path.exists(preproc_path):
            pre = load_pickle(preproc_path)
            Xt = pre.transform(Xf)
            return Xt
        if os.path.exists(scaler_path):
            scaler = load_pickle(scaler_path)
            Xf = scaler.transform(Xf)
        if os.path.exists(encoder_path):
            enc = load_pickle(encoder_path)
            try:
                Xf = enc.transform(Xf)
            except Exception:
                # If encoder expects categorical columns only, skip here
                pass
    except Exception as e:
        st.warning(f"Preprocessing failed: {e}")

    return Xf.values if hasattr(Xf, 'values') else Xf


def predict_with_model(model, X):
    # Try predict_proba
    try:
        proba = model.predict_proba(X)
        if proba.shape[1] == 2:
            fraud_proba = proba[:, 1]
        else:
            # multiclass? take max of last
            fraud_proba = proba.max(axis=1)
        preds = (fraud_proba >= 0.5).astype(int)
        return preds, fraud_proba
    except Exception:
        try:
            preds = model.predict(X)
            # If predict gives 0/1, assign probability 1 for predicted class
            proba = np.where(preds == 1, 1.0, 0.0)
            return preds, proba
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
            return None, None


def risk_category(p: float) -> str:
    if p <= 0.30:
        return "Low Risk"
    if p <= 0.60:
        return "Medium Risk"
    if p <= 0.85:
        return "High Risk"
    return "Critical Risk"


def suggested_action(category: str) -> str:
    return {
        "Low Risk": "Approve",
        "Medium Risk": "Monitor",
        "High Risk": "Flag for Review",
        "Critical Risk": "Block / Escalate",
    }.get(category, "Monitor")


def main():
    st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

    st.title("Credit Card Fraud Detection")

    st.sidebar.header("Model & Files")
    models = get_available_models()
    model_choice = st.sidebar.selectbox("Select model file", options=["-- Select --"] + models)
    model_obj = None
    if model_choice and model_choice != "-- Select --":
        model_path = os.path.join(MODELS_DIR, model_choice)
        model_obj = load_model(model_path)

    uploaded_file = st.file_uploader("Upload transactions (CSV or Excel)", type=["csv", "xlsx", "xls"])

    df: Optional[pd.DataFrame] = None
    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success(f"Loaded {uploaded_file.name} — {df.shape[0]} rows")
        except Exception as e:
            st.error(f"Failed to read file: {e}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("Place your trained models and preprocessing files in the `models/` folder.")
    st.sidebar.markdown("Supported preprocessing files: `preprocessor.pkl`, `scaler.pkl`, `encoder.pkl`, `feature_columns.pkl`.")

    if df is not None:
        st.subheader("Data Preview")
        st.dataframe(df.head(100))

        # Determine feature columns
        feature_cols = None
        feature_cols_path = os.path.join(MODELS_DIR, 'feature_columns.pkl')
        if os.path.exists(feature_cols_path):
            try:
                feature_cols = load_pickle(feature_cols_path)
                st.info(f"Using feature columns from feature_columns.pkl ({len(feature_cols)} columns)")
            except Exception as e:
                st.warning(f"Could not load feature_columns.pkl: {e}")

        if feature_cols is None and model_obj is not None:
            # try attribute
            try:
                if hasattr(model_obj, 'feature_names_in_'):
                    feature_cols = list(model_obj.feature_names_in_)
                elif hasattr(model_obj, 'n_features_in_'):
                    # fallback: use first n columns
                    n = int(model_obj.n_features_in_)
                    feature_cols = list(df.columns[:n])
            except Exception:
                feature_cols = None

        if feature_cols is None:
            st.warning("Feature columns not detected. Please select input features from your uploaded data.")
            feature_cols = st.multiselect("Select feature columns to use for prediction", options=list(df.columns), default=list(df.columns))

        if not feature_cols:
            st.error("No feature columns selected — cannot predict.")
            return

        st.markdown(f"**Using {len(feature_cols)} features**")

        # Prepare data for prediction
        X_ready = apply_preprocessors(df, feature_cols)

        if model_obj is None:
            st.warning("No model selected — select a model from the sidebar to run predictions.")
        else:
            st.subheader("Predictions")
            preds, probs = predict_with_model(model_obj, X_ready)
            if preds is None:
                st.error("Prediction failed.")
                return

            results = df.copy()
            results['predicted_label'] = preds
            results['fraud_probability'] = probs
            results['risk_category'] = results['fraud_probability'].apply(risk_category)
            results['suggested_action'] = results['risk_category'].apply(suggested_action)

            st.dataframe(results.head(100))

            # Summary metrics
            total = len(results)
            fraud_count = int(results['predicted_label'].sum())
            fraud_pct = fraud_count / total * 100 if total > 0 else 0
            high_critical = results[results['risk_category'].isin(['High Risk', 'Critical Risk'])].shape[0]
            critical_count = results[results['risk_category'] == 'Critical Risk'].shape[0]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total transactions", total)
            col2.metric("Predicted fraudulent", fraud_count)
            col3.metric("Fraud percentage", f"{fraud_pct:.2f}%")
            col4.metric("High+Critical", high_critical)

            # Fraud by amount
            if 'Amount' in results.columns:
                fig_amount = px.histogram(results, x='Amount', color='predicted_label', nbins=50, title='Fraud by Transaction Amount')
                st.plotly_chart(fig_amount, use_container_width=True)

            # Fraud by merchant category
            merchant_col = None
            for c in ['merchant_category', 'merchant', 'category', 'MCC']:
                if c in results.columns:
                    merchant_col = c
                    break
            if merchant_col:
                mdf = results.groupby(merchant_col)['predicted_label'].sum().reset_index().sort_values('predicted_label', ascending=False).head(20)
                fig_mc = px.bar(mdf, x=merchant_col, y='predicted_label', title='Fraud by Merchant Category')
                st.plotly_chart(fig_mc, use_container_width=True)

            # Fraud by location
            loc_col = None
            for c in ['location', 'country', 'city', 'state']:
                if c in results.columns:
                    loc_col = c
                    break
            if loc_col:
                ldf = results.groupby(loc_col)['predicted_label'].sum().reset_index().sort_values('predicted_label', ascending=False).head(20)
                fig_loc = px.bar(ldf, x=loc_col, y='predicted_label', title='Fraud by Location')
                st.plotly_chart(fig_loc, use_container_width=True)

            # Top 10 risky transactions
            top10 = results.sort_values('fraud_probability', ascending=False).head(10)
            st.subheader("Top 10 Risky Transactions")
            st.dataframe(top10)

            # Downloads
            csv = results.to_csv(index=False)
            st.download_button("Download full results (CSV)", data=csv, file_name="predictions.csv", mime='text/csv')

            high_risk_df = results[results['risk_category'].isin(['High Risk', 'Critical Risk'])]
            if not high_risk_df.empty:
                csv_hr = high_risk_df.to_csv(index=False)
                st.download_button("Download high-risk transactions (CSV)", data=csv_hr, file_name="high_risk_transactions.csv", mime='text/csv')

            # Summary Excel
            try:
                towrite = io.BytesIO()
                with pd.ExcelWriter(towrite, engine='openpyxl') as writer:
                    results.to_excel(writer, sheet_name='predictions', index=False)
                    summary = pd.DataFrame({
                        'total_transactions': [total],
                        'predicted_fraudulent': [fraud_count],
                        'fraud_percentage': [fraud_pct],
                        'high_critical': [high_critical],
                        'critical_count': [critical_count]
                    })
                    summary.to_excel(writer, sheet_name='summary', index=False)
                towrite.seek(0)
                st.download_button("Download summary report (Excel)", data=towrite, file_name="fraud_summary.xlsx", mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            except Exception as e:
                st.info("Excel export requires openpyxl. Install via requirements.txt if needed.")


if __name__ == '__main__':
    main()
