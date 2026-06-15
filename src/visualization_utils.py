import matplotlib
# Use non-interactive backend to avoid Tkinter errors when running in
# background threads or headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay
)


def plot_confusion_matrix(model, X_test, y_test, save_path):

    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test
    )

    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(y_test, y_prob, save_path):

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure(figsize=(6, 4))

    plt.plot(fpr, tpr)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.savefig(save_path)
    plt.close()


def plot_precision_recall_curve(
    y_test,
    y_prob,
    save_path
):

    precision, recall, _ = precision_recall_curve(
        y_test,
        y_prob
    )

    plt.figure(figsize=(6, 4))

    plt.plot(
        recall,
        precision
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve")

    plt.savefig(save_path)

    plt.close()


def plot_dnn_training_curve(
    history,
    save_path
):

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.legend()

    plt.title("DNN Loss Curve")

    plt.savefig(save_path)

    plt.close()