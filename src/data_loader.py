# src/data_loader.py

import pandas as pd


def load_data(file_path):
    """
    Load credit card fraud dataset

    Parameters
    ----------
    file_path : str
        Path to CSV file

    Returns
    -------
    pd.DataFrame
    """

    print("=" * 50)
    print("Loading Dataset")
    print("=" * 50)

    df = pd.read_csv(file_path)

    print(f"Dataset Shape : {df.shape}")
    print(f"Rows          : {df.shape[0]}")
    print(f"Columns       : {df.shape[1]}")

    return df