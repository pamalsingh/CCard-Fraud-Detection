import pandas as pd


def save_error_analysis(
    X_test,
    y_test,
    y_pred,
    y_prob,
    output_dir,
    model_name
):

    results = X_test.copy()

    results["Actual"] = y_test.values
    results["Predicted"] = y_pred
    results["Probability"] = y_prob

    false_positive = results[
        (results["Actual"] == 0)
        &
        (results["Predicted"] == 1)
    ]

    false_negative = results[
        (results["Actual"] == 1)
        &
        (results["Predicted"] == 0)
    ]

    false_positive.to_csv(
        f"{output_dir}/{model_name}_false_positive.csv",
        index=False
    )

    false_negative.to_csv(
        f"{output_dir}/{model_name}_false_negative.csv",
        index=False
    )

    results.head(50).to_csv(
        f"{output_dir}/{model_name}_sample_predictions.csv",
        index=False
    )

    # Save a small set of failure examples for quick inspection
    results.head(10).to_csv(
        f"{output_dir}/{model_name}_failure_examples.csv",
        index=False
    )