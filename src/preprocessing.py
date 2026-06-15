# src/preprocessing.py

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler


def preprocess_data(df):

    print("=" * 50)
    print("Preprocessing Data")
    print("=" * 50)
    
    # Drop duplicate rows
    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    rows_after_dropping = df.shape[0]
    print(f"Dropped {initial_rows - rows_after_dropping} duplicate rows.")
    print(f"Remaining rows: {rows_after_dropping}")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    amount_scaler = RobustScaler()

    X_train["Amount"] = amount_scaler.fit_transform(
        X_train[["Amount"]]
    )

    X_test["Amount"] = amount_scaler.transform(
        X_test[["Amount"]]
    )

    feature_scaler = StandardScaler()

    columns_to_scale = [
        col for col in X_train.columns
        if col != "Amount"
    ]

    X_train[columns_to_scale] = feature_scaler.fit_transform(
        X_train[columns_to_scale]
    )

    X_test[columns_to_scale] = feature_scaler.transform(
        X_test[columns_to_scale]
    )

    print(f"Training Shape : {X_train.shape}")
    print(f"Testing Shape  : {X_test.shape}")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        amount_scaler,
        feature_scaler
    )