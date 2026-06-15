# src/experiment_tracker.py

import os
import json
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support
)
from sklearn.metrics import precision_recall_curve, auc


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_classification_report(report_text, out_path):
    with open(out_path, "w") as f:
        f.write(report_text)


def plot_confusion_matrix(cm, labels, out_path):
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def record_experiment(
    name,
    y_true,
    y_pred,
    y_prob,
    out_dir,
    smote_used=False,
    hyperparameters=None,
    inference_latency_ms=None
):
    ensure_dir(out_dir)
    # classification report
    report_text = classification_report(y_true, y_pred)
    save_classification_report(report_text, os.path.join(out_dir, f"{name}_classification_report.txt"))

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, labels=[0, 1], out_path=os.path.join(out_dir, f"{name}_confusion_matrix.png"))

    # ROC AUC (expect valid probability vector)
    roc = roc_auc_score(y_true, y_prob)

    # PR-AUC (precision-recall AUC)
    precision_curve, recall_curve, _ = precision_recall_curve(
        y_true,
        y_prob
    )

    pr_auc = auc(
        recall_curve,
        precision_curve
    )

    # precision/recall/f1 for binary case
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

    # count of positive samples in the true labels
    support_pos = int(np.sum(np.array(y_true) == 1))

    summary = {
        "model": name,
        "smote_used": smote_used,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc),
        "pr_auc": float(pr_auc),
        "support_pos": int(support_pos),
        "hyperparameters": str(hyperparameters),
        "inference_latency_ms": inference_latency_ms,
        "run_timestamp": datetime.utcnow().isoformat()
    }

    df = pd.DataFrame([summary])
    csv_path = os.path.join(out_dir, "experiments_summary.csv")
    # append row to CSV in a safe way: read existing file (if any), concat, and rewrite
    try:
        if os.path.exists(csv_path):
            existing = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
            combined = pd.concat([existing, df], ignore_index=True)
            # keep only the latest run per model+smote+hyperparameters to avoid duplicates
            if {"model", "smote_used", "hyperparameters"}.issubset(combined.columns):
                combined = combined.sort_values(by="run_timestamp").drop_duplicates(
                    subset=["model", "smote_used", "hyperparameters"],
                    keep="last"
                )
            combined.to_csv(csv_path, index=False)
        else:
            os.makedirs(out_dir, exist_ok=True)
            df.to_csv(csv_path, index=False)
    except Exception:
        # fallback to simple append if something goes wrong
        df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)

    # also save JSON summary
    with open(os.path.join(out_dir, f"{name}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
