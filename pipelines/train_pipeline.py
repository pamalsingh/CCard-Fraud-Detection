# pipelines/train_pipeline.py

import sys
from pathlib import Path

# Ensure project root is on sys.path so `from src...` imports work
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_data
from src.eda import generate_eda_report
from src.preprocessing import preprocess_data
from src.imbalance_handler import apply_smote
# timing for inference latency measurements
import time

from src.train_rf import train_random_forest
# tuner is kept but disabled in pipeline for now
# from src.tune_rf import tune_random_forest
# Logistic Regression steps are disabled for now
# from src.train_lr import train_logistic_regression
from src.experiment_tracker import record_experiment
from src.train_dnn import train_dnn
from sklearn.model_selection import train_test_split
from src.evaluate import evaluate_sklearn_model
from src.train_xgb import train_xgboost
from src.save_artifacts import (
    save_random_forest,
    save_scaler,
    save_model,
    save_keras_model
)
from src import visualization_utils as viz
from src.error_analysis import save_error_analysis
from src.comparison_report import generate_comparison_report
import json
from pathlib import Path

# No runtime toggles: pipeline runs LR, RF, XGBoost (if available), and DNN unconditionally


def run_training_pipeline(tuning_file: str = None):
    print("=" * 70)
    print("CREDIT CARD FRAUD DETECTION TRAINING")
    print("=" * 70)

    # ------------------------------------------------
    # STEP 1
    # Load Dataset
    # ------------------------------------------------

    df = load_data(
        "data/raw/creditcard.csv"
    )

    # ------------------------------------------------
    # STEP 2
    # EDA
    # ------------------------------------------------

    generate_eda_report(df)

    # ------------------------------------------------
    # STEP 3
    # Preprocessing
    # ------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        amount_scaler,
        feature_scaler
    ) = preprocess_data(df)

    # Attempt to load tuning results (optional). If present, these will be
    # used to pass tuned hyperparameters into the training functions.
    tuning_path = Path(tuning_file) if tuning_file else Path("reports/tuning_results.json")
    tuning_results = {}
    if tuning_path.exists():
        try:
            with tuning_path.open("r") as fh:
                tuning_results = json.load(fh)
            print(f"Loaded tuning results from {tuning_path}")
        except Exception as e:
            print("Failed to read tuning results, continuing with defaults:", e)

    def _get_best_params(mname: str):
        """Resolve best parameter dict for model name from various tuning JSON shapes."""
        # Case A: file is a mapping with top-level keys 'rf','xgb','dnn'
        val = tuning_results.get(mname)
        if isinstance(val, dict):
            # If val itself contains 'best_params', unwrap
            if "best_params" in val:
                return val.get("best_params")
            return val

        # Case B: file has shape {"best_params": { 'rf': {...}, ... }, ... }
        bp = tuning_results.get("best_params")
        if isinstance(bp, dict):
            inner = bp.get(mname)
            if isinstance(inner, dict):
                if "best_params" in inner:
                    return inner.get("best_params")
                return inner

        return None

    # ------------------------------------------------
    # STEP 4a
    # Pre-SMOTE training (baseline on original training data)
    # ------------------------------------------------

    rf_model_pre = train_random_forest(
        X_train,
        y_train,
        params=_get_best_params("rf")
    )

    # Evaluate RF baseline
    start = time.time()
    y_prob_rf_pre = rf_model_pre.predict_proba(X_test)[:, 1]
    end = time.time()
    latency_ms = (end - start) * 1000.0 / max(1, len(X_test))
    y_pred_rf_pre = rf_model_pre.predict(X_test)
    record_experiment(
        "random_forest_pre_smote",
        y_test,
        y_pred_rf_pre,
        y_prob_rf_pre,
        out_dir="reports",
        smote_used=False,
        inference_latency_ms=float(latency_ms)
    )

    # Try XGBoost baseline (if available)
    try:
        xgb_model_pre = train_xgboost(X_train, y_train, params=_get_best_params("xgb"))
        start = time.time()
        y_prob_xgb_pre = xgb_model_pre.predict_proba(X_test)[:, 1]
        end = time.time()
        latency_ms = (end - start) * 1000.0 / max(1, len(X_test))
        y_pred_xgb_pre = xgb_model_pre.predict(X_test)
        record_experiment(
            "xgboost_pre_smote",
            y_test,
            y_pred_xgb_pre,
            y_prob_xgb_pre,
            out_dir="reports",
            smote_used=False,
            inference_latency_ms=float(latency_ms)
        )
        # plots and error analysis for XGBoost pre-SMOTE
        viz.plot_roc_curve(y_test, y_prob_xgb_pre, "reports/xgboost_pre_smote_roc.png")
        viz.plot_precision_recall_curve(y_test, y_prob_xgb_pre, "reports/xgboost_pre_smote_pr.png")
        save_error_analysis(X_test, y_test, y_pred_xgb_pre, y_prob_xgb_pre, "reports", "xgboost_pre_smote")
    except Exception as e:
        print("XGBoost pre-SMOTE training failed or unavailable:", e)

    # Train DNN pre-SMOTE (baseline)
    X_tr_pre, X_val_pre, y_tr_pre, y_val_pre = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=42,
        stratify=y_train
    )

    _dnn_params_pre = _get_best_params("dnn") or {}
    dnn_model_pre, history_pre = train_dnn(
        X_tr_pre,
        y_tr_pre,
        X_val_pre,
        y_val_pre,
        learning_rate=_dnn_params_pre.get("learning_rate", 0.001),
        dropout_rate=_dnn_params_pre.get("dropout_rate", 0.30),
        batch_size=_dnn_params_pre.get("batch_size", 256),
        epochs=_dnn_params_pre.get("epochs", 50)
    )

    start = time.time()
    y_prob_dnn_pre = dnn_model_pre.predict(X_test).ravel()
    end = time.time()
    latency_ms = (end - start) * 1000.0 / max(1, len(X_test))
    y_pred_dnn_pre = (y_prob_dnn_pre >= 0.50).astype(int)
    record_experiment(
        "dnn_pre_smote",
        y_test,
        y_pred_dnn_pre,
        y_prob_dnn_pre,
        out_dir="reports",
        smote_used=False,
        inference_latency_ms=float(latency_ms)
    )
    viz.plot_roc_curve(y_test, y_prob_dnn_pre, "reports/dnn_pre_smote_roc.png")
    viz.plot_precision_recall_curve(y_test, y_prob_dnn_pre, "reports/dnn_pre_smote_pr.png")
    viz.plot_dnn_training_curve(history_pre, "reports/dnn_pre_smote_loss.png")
    save_error_analysis(X_test, y_test, y_pred_dnn_pre, y_prob_dnn_pre, "reports", "dnn_pre_smote")

    # ------------------------------------------------
    # STEP 4
    # SMOTE (applied to training fold)
    # ------------------------------------------------

    X_train_smote, y_train_smote = apply_smote(
        X_train,
        y_train
    )

    # ------------------------------------------------
    # STEP 5
    # Tune Random Forest (GridSearch) on the (resampled) training set,
    # then use the best estimator for evaluation and saving.
    # ------------------------------------------------

    # Logistic Regression disabled for now (kept for reference)
    # lr_model = train_logistic_regression(X_train_smote, y_train_smote)
    # y_prob_lr = lr_model.predict_proba(X_test)[:, 1]
    # y_pred_lr = lr_model.predict(X_test)
    # record_experiment("logistic_regression", y_test, y_pred_lr, y_prob_lr, out_dir="reports")

    # Train Random Forest on SMOTE-resampled data
    rf_model = train_random_forest(
        X_train_smote,
        y_train_smote,
        params=_get_best_params("rf")
    )

    # Evaluate RF (post-SMOTE)
    start = time.time()
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    end = time.time()
    latency_ms = (end - start) * 1000.0 / max(1, len(X_test))
    y_pred_rf = rf_model.predict(X_test)
    record_experiment(
        "random_forest_post_smote",
        y_test,
        y_pred_rf,
        y_prob_rf,
        out_dir="reports",
        smote_used=True,
        hyperparameters=rf_model.get_params(),
        inference_latency_ms=float(latency_ms)
    )
    viz.plot_roc_curve(y_test, y_prob_rf, "reports/random_forest_post_smote_roc.png")
    viz.plot_precision_recall_curve(y_test, y_prob_rf, "reports/random_forest_post_smote_pr.png")
    save_error_analysis(X_test, y_test, y_pred_rf, y_prob_rf, "reports", "random_forest_post_smote")

    # Train XGBoost (if available)
    try:
        xgb_model = train_xgboost(X_train_smote, y_train_smote, params=_get_best_params("xgb"))
        start = time.time()
        y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
        end = time.time()
        latency_ms = (end - start) * 1000.0 / max(1, len(X_test))
        y_pred_xgb = xgb_model.predict(X_test)
        record_experiment(
            "xgboost_post_smote",
            y_test,
            y_pred_xgb,
            y_prob_xgb,
            out_dir="reports",
            smote_used=True,
            inference_latency_ms=float(latency_ms)
        )
        viz.plot_roc_curve(y_test, y_prob_xgb, "reports/xgboost_post_smote_roc.png")
        viz.plot_precision_recall_curve(y_test, y_prob_xgb, "reports/xgboost_post_smote_pr.png")
        save_error_analysis(X_test, y_test, y_pred_xgb, y_prob_xgb, "reports", "xgboost_post_smote")
    except ImportError as ie:
        print("XGBoost not available, skipping XGBoost training:", ie)
    except Exception as e:
        print("XGBoost training failed, continuing:", e)

    # ------------------------------------------------
    # STEP 6
    # Train DNN (uses class weights)
    # ------------------------------------------------
    # create a stratified validation split from the training set for DNN
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_smote,
        y_train_smote,
        test_size=0.20,
        random_state=42,
        stratify=y_train_smote
    )

    _dnn_params = _get_best_params("dnn") or {}
    dnn_model, history = train_dnn(
        X_tr,
        y_tr,
        X_val,
        y_val,
        learning_rate=_dnn_params.get("learning_rate", 0.001),
        dropout_rate=_dnn_params.get("dropout_rate", 0.30),
        batch_size=_dnn_params.get("batch_size", 256),
        epochs=_dnn_params.get("epochs", 50)
    )

    # DNN post-SMOTE evaluation and artifacts
    start = time.time()
    y_prob_dnn = dnn_model.predict(X_test).ravel()
    end = time.time()
    latency_ms = (end - start) * 1000.0 / max(1, len(X_test))
    y_pred_dnn = (y_prob_dnn >= 0.50).astype(int)
    record_experiment(
        "dnn_post_smote",
        y_test,
        y_pred_dnn,
        y_prob_dnn,
        out_dir="reports",
        smote_used=True,
        inference_latency_ms=float(latency_ms)
    )
    viz.plot_roc_curve(y_test, y_prob_dnn, "reports/dnn_post_smote_roc.png")
    viz.plot_precision_recall_curve(y_test, y_prob_dnn, "reports/dnn_post_smote_pr.png")
    viz.plot_dnn_training_curve(history, "reports/dnn_post_smote_loss.png")
    save_error_analysis(X_test, y_test, y_pred_dnn, y_prob_dnn, "reports", "dnn_post_smote")

    # ------------------------------------------------
    # STEP 7
    # Evaluate RF
    # ------------------------------------------------
    evaluate_sklearn_model(
        rf_model,
        X_test,
        y_test
    )

    # ------------------------------------------------
    # STEP 8
    # Save Artifacts
    # ------------------------------------------------

    # Save both pre- and post-SMOTE RF models for inspection
    save_random_forest(
        rf_model_pre,
        "models/rf_model_pre_smote.pkl"
    )

    save_random_forest(
        rf_model,
        "models/rf_model_post_smote.pkl"
    )

    # Save XGBoost models if trained
    try:
        save_model(xgb_model_pre, "models/xgb_model_pre_smote.pkl")
    except Exception:
        pass

    try:
        save_model(xgb_model, "models/xgb_model_post_smote.pkl")
    except Exception:
        pass

    # Save DNN models
    try:
        save_keras_model(dnn_model_pre, "models/dnn_pre_smote")
    except Exception:
        pass

    try:
        save_keras_model(dnn_model, "models/dnn_post_smote")
    except Exception:
        pass

    save_scaler(
        amount_scaler,
        "models/amount_scaler.pkl"
    )

    save_scaler(
        feature_scaler,
        "models/feature_scaler.pkl"
    )

    # Generate comparison report and model list
    generate_comparison_report()

    print("=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning-file", default=None, help="Path to tuning_results.json to apply tuned hyperparameters")
    args = parser.parse_args()

    run_training_pipeline(tuning_file=args.tuning_file)
