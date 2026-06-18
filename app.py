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
    # Use joblib.load first (restores sklearn objects reliably), fall back to pickle
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def load_model(path: str):
    # Prefer Keras/TensorFlow loader for .h5/.keras/.hdf5 files or SavedModel dirs
    try:
        if os.path.isdir(path) or path.lower().endswith(('.h5', '.keras', '.hdf5')):
            try:
                from tensorflow.keras.models import load_model as _keras_load
            except Exception as e:
                st.warning(f"TensorFlow/Keras not available: {e}")
            else:
                try:
                    return _keras_load(path)
                except Exception as e:
                    st.warning(f"Keras load failed for {path}: {e}")

        # Fallback to joblib / pickle for sklearn-like models
        try:
            return joblib.load(path)
        except Exception:
            with open(path, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load model {path}: {e}")
        return None


def get_available_models():
    if not os.path.isdir(MODELS_DIR):
        return []
    entries = os.listdir(MODELS_DIR)
    model_files = []
    for e in entries:
        full = os.path.join(MODELS_DIR, e)
        if os.path.isfile(full) and e.lower().endswith(('.pkl', '.joblib', '.h5', '.keras', '.hdf5')):
            model_files.append(e)
        # detect TF SavedModel directory by presence of saved_model.pb
        elif os.path.isdir(full) and os.path.exists(os.path.join(full, "saved_model.pb")):
            model_files.append(e)
    return sorted(model_files)


def apply_preprocessors(df: pd.DataFrame, feature_cols, models_folder=MODELS_DIR):
    X = df.copy()
    # Select features
    Xf = X.loc[:, feature_cols].copy()

    # Apply preprocessor if exists
    preproc_path = os.path.join(models_folder, 'preprocessor.pkl')
    scaler_path = os.path.join(models_folder, 'scaler.pkl')
    # pipeline-compatible scaler names
    amount_scaler_path = os.path.join(models_folder, 'amount_scaler.pkl')
    feature_scaler_path = os.path.join(models_folder, 'feature_scaler.pkl')

    try:
        if os.path.exists(preproc_path):
            pre = load_pickle(preproc_path)
            # If preprocessor file contains feature column list, apply ordering
            if isinstance(pre, (list, tuple, np.ndarray)):
                cols = list(pre)
                missing = [c for c in cols if c not in Xf.columns]
                if not missing:
                    Xf = Xf[cols]
                    return Xf
                else:
                    st.warning(f"preprocessor.pkl looks like feature columns but missing: {missing}; continuing")
            elif hasattr(pre, 'transform'):
                Xt = pre.transform(Xf)
                return Xt
            else:
                st.warning("preprocessor.pkl is not a transformer; skipping")
        # Support older pipeline output names: apply amount_scaler and feature_scaler
        if os.path.exists(amount_scaler_path):
            amt_scaler = load_pickle(amount_scaler_path)
            if hasattr(amt_scaler, 'transform'):
                if 'Amount' in Xf.columns:
                    Xf['Amount'] = amt_scaler.transform(Xf[['Amount']])
            else:
                st.warning('Loaded amount_scaler.pkl does not have transform(); skipping')

        if os.path.exists(feature_scaler_path):
            feat_scaler = load_pickle(feature_scaler_path)
            if hasattr(feat_scaler, 'transform'):
                cols = [c for c in Xf.columns if c != 'Amount']
                if cols:
                    Xf[cols] = feat_scaler.transform(Xf[cols])
            else:
                st.warning('Loaded feature_scaler.pkl does not have transform(); skipping')

        # backward-compatible single scaler name
        if os.path.exists(scaler_path):
            scaler = load_pickle(scaler_path)
            if hasattr(scaler, 'transform'):
                Xf = scaler.transform(Xf)
            else:
                st.warning('Loaded scaler.pkl does not have transform(); skipping')
        # Note: encoder.pkl support removed — encode categorical features during training
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
            raw = model.predict(X)
            # Normalize to 1d array
            arr = np.array(raw).ravel()
            # If outputs are not probabilities (e.g. logits), apply sigmoid
            if arr.min() < 0 or arr.max() > 1:
                probs = 1.0 / (1.0 + np.exp(-arr))
            else:
                probs = arr
            preds = (probs >= 0.5).astype(int)
            return preds, probs
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

    # Debug helper: show which preprocessor files the running app actually sees
    debug = st.sidebar.checkbox("Debug: show loaded preprocessors")
    if debug:
        st.sidebar.markdown("**Preprocessor debug**")
        files_to_check = [
            'feature_columns.pkl', 'preprocessor.pkl', 'scaler.pkl',
            'amount_scaler.pkl', 'feature_scaler.pkl', 'encoder.pkl'
        ]
        for fname in files_to_check:
            p = os.path.join(MODELS_DIR, fname)
            exists = os.path.exists(p)
            st.sidebar.write(f"{fname}: exists={exists} — {os.path.abspath(p)}")
            if exists:
                try:
                    obj = load_pickle(p)
                    st.sidebar.write(f"  type={type(obj)}; has_transform={hasattr(obj,'transform')}")
                except Exception as e:
                    st.sidebar.write(f"  failed to load: {e}")

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

        # If user accidentally included the target column 'Class' or similar, remove it
        removed_cols = []
        for col_to_drop in ['Class', 'class', 'label', 'target']:
            if col_to_drop in feature_cols:
                feature_cols = [c for c in feature_cols if c != col_to_drop]
                removed_cols.append(col_to_drop)
        if removed_cols:
            st.warning(f"Removed target column(s) from features: {', '.join(removed_cols)}")

        if not feature_cols:
            st.error("No feature columns selected — cannot predict.")
            return

        st.markdown(f"**Using {len(feature_cols)} features**")

        # Prepare data for prediction
        # Validate feature count for Keras models
        if model_obj is not None:
            try:
                # Keras models expose input_shape; sklearn models have n_features_in_
                expected = None
                if hasattr(model_obj, 'input_shape'):
                    expected = int(model_obj.input_shape[-1]) if model_obj.input_shape is not None else None
                elif hasattr(model_obj, 'n_features_in_'):
                    expected = int(model_obj.n_features_in_)

                if expected is not None and len(feature_cols) != expected:
                    st.error(
                        f"Model expects {expected} features but you selected {len(feature_cols)}. "
                        "Ensure you exclude the target column and use the same feature order used during training, or place a `feature_columns.pkl` file in the `models/` folder."
                    )
                    return
            except Exception:
                # If any inspection fails, continue and let prediction handle shape errors
                pass

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
