# ============================================================
# Credit Card Fraud Detection Project
# File: eda.py
#
# Purpose:
# Perform Exploratory Data Analysis (EDA)
#
# Business Goal:
# Understand fraud distribution, transaction patterns,
# data quality and class imbalance before model training.
# ============================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import os


# Helper: save figures to reports/ when using non-interactive backends (e.g. Agg)
_EDA_FIG_IDX = 0
def _save_or_show(name: str = None):
    """Save the current matplotlib figure to reports/ if backend is non-interactive,
    otherwise show it. Filenames are generated automatically when `name` is None.
    """
    global _EDA_FIG_IDX
    backend = matplotlib.get_backend().lower()
    os.makedirs("reports", exist_ok=True)
    if name is None:
        name = f"eda_{_EDA_FIG_IDX}"
    filename = os.path.join("reports", f"{name}.png")
    try:
        if "agg" in backend:
            plt.savefig(filename, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
    finally:
        _EDA_FIG_IDX += 1


def basic_data_overview(df):
    """
    Display basic dataset information.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    None
    """

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nData Types & Null Values:")
    df.info()

    print("\nSummary Statistics:")
    print(df.describe())


def check_missing_values(df):
    """
    Check missing values in dataset.
    """

    print("\n" + "=" * 60)
    print("MISSING VALUE ANALYSIS")
    print("=" * 60)

    missing_values = df.isnull().sum()

    print(missing_values)

    print("\nTotal Missing Values:")
    print(missing_values.sum())


def check_duplicates(df):
    """
    Check duplicate rows.
    """

    print("\n" + "=" * 60)
    print("DUPLICATE RECORD ANALYSIS")
    print("=" * 60)

    duplicate_count = df.duplicated().sum()

    print(f"Duplicate Rows : {duplicate_count}")


def class_distribution_analysis(df):
    """
    Analyze fraud vs non-fraud distribution.
    """

    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("=" * 60)

    class_counts = df["Class"].value_counts()

    class_percentage = (
        df["Class"]
        .value_counts(normalize=True)
        * 100
    )

    print("\nClass Counts:")
    print(class_counts)

    print("\nClass Percentage:")
    print(class_percentage)

    plt.figure(figsize=(8, 5))

    sns.countplot(
        x="Class",
        data=df
    )

    plt.title("Fraud vs Non-Fraud Transactions")
    plt.xlabel("Class")
    plt.ylabel("Count")

    _save_or_show("class_distribution")

    return class_counts, class_percentage


def amount_analysis(df):
    """
    Analyze transaction amount behaviour.
    """

    print("\n" + "=" * 60)
    print("TRANSACTION AMOUNT ANALYSIS")
    print("=" * 60)

    print(df["Amount"].describe())

    print("\nAmount Skewness:")
    print(df["Amount"].skew())

    # ---------------------------------------------------
    # Original Amount Distribution
    # ---------------------------------------------------

    plt.figure(figsize=(10, 5))

    sns.histplot(
        df["Amount"],
        bins=100,
        kde=True
    )

    plt.title("Transaction Amount Distribution")

    _save_or_show("amount_distribution")

    # ---------------------------------------------------
    # Log Transformed Amount
    # ---------------------------------------------------

    plt.figure(figsize=(10, 5))

    sns.histplot(
        np.log1p(df["Amount"]),
        bins=100,
        kde=True
    )

    plt.title("Log Transformed Amount Distribution")

    _save_or_show("amount_log_distribution")

    # ---------------------------------------------------
    # Fraud vs Non-Fraud Amount Comparison
    # ---------------------------------------------------

    plt.figure(figsize=(10, 5))

    sns.boxplot(
        x="Class",
        y="Amount",
        data=df
    )

    plt.title(
        "Transaction Amount by Fraud Class"
    )

    _save_or_show("amount_by_class")


def correlation_analysis(df):
    """
    Generate correlation heatmap.
    """

    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)

    plt.figure(figsize=(16, 12))

    correlation_matrix = df.corr()

    sns.heatmap(
        correlation_matrix,
        cmap="coolwarm",
        center=0
    )

    plt.title("Feature Correlation Heatmap")

    _save_or_show("correlation_heatmap")


def generate_eda_report(df):
    """
    Run complete EDA workflow.
    """

    print("\n")
    print("=" * 70)
    print("STARTING EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    basic_data_overview(df)

    check_missing_values(df)

    check_duplicates(df)

    class_distribution_analysis(df)

    amount_analysis(df)

    correlation_analysis(df)

    print("\n")
    print("=" * 70)
    print("EDA COMPLETED SUCCESSFULLY")
    print("=" * 70)