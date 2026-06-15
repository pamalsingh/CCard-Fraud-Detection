# src/evaluate.py

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)


def evaluate_sklearn_model(
    model,
    X_test,
    y_test
):

    y_prob = model.predict_proba(X_test)[:, 1]

    y_pred = (y_prob >= 0.50).astype(int)

    print("=" * 50)
    print("Model Evaluation")
    print("=" * 50)

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    print(
        "ROC AUC :",
        roc_auc_score(
            y_test,
            y_prob
        )
    )

    print(
        "Confusion Matrix"
    )

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )