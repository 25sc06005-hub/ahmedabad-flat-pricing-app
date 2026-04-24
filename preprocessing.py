import pandas as pd
import numpy as np




def clean_data(df):
    """
    Standard cleaning used for both training and inference.
    """

    df = df.copy()

    # ----------------------------
    # Location cleaning (SAFE ORDER)
    # ----------------------------
    if "location" in df.columns:
        df["location"] = df["location"].fillna("Other").astype(str)

    # ----------------------------
    # Numeric conversion safety
    # ----------------------------
    for col in ["price", "price_sqft", "total_sqft", "bhk"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ----------------------------
    # Drop rows with critical missing values (TRAINING SAFETY)
    # ----------------------------
    if "total_sqft" in df.columns:
        df = df.dropna(subset=["total_sqft"])

    return df




def get_features_target(df):
    """
    Splits features and target safely, avoiding leakage.
    """

    target = "price"

    if target not in df.columns:
        return df, None

    drop_cols = [
        "price",
        "price_sqft",
        "price_segment"
    ]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[target]

    return X, y