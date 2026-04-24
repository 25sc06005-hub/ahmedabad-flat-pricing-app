import pandas as pd
import numpy as np




def clean_data(df):

    """Standard cleaning used by both training and inference."""

    df = df.copy()
    df["location"] = df["location"].astype(str).fillna("Other")
    
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    if "price_sqft" in df.columns:
        df["price_sqft"] = pd.to_numeric(df["price_sqft"], errors="coerce")
    
    # Drop rows missing crucial training info
    df = df.dropna(subset=["total_sqft"])
    return df




def get_features_target(df):

    """Separates features from target and removes leakage."""
    target = "price"

    # Explicitly drop leakage and non-feature columns
    drop_cols = [target, "price_sqft", "price_segment"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[target] if target in df.columns else None
    return X, y