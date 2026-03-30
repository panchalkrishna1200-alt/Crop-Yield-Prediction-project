"""
data_loader.py
--------------
Handles loading, cleaning, and preprocessing the Kaggle Crop Production dataset.
Expected columns (Kaggle - Crop Production in India):
  State_Name, District_Name, Crop_Year, Season, Crop, Area, Production
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. Load raw data
# ─────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """Load the CSV from disk. Accepts absolute or relative path."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df


# ─────────────────────────────────────────────
# 2. Basic info helper
# ─────────────────────────────────────────────
def dataset_info(df: pd.DataFrame) -> dict:
    """Return a summary dict for display in Streamlit."""
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "states": df["State_Name"].nunique() if "State_Name" in df.columns else "N/A",
        "crops": df["Crop"].nunique() if "Crop" in df.columns else "N/A",
        "seasons": df["Season"].nunique() if "Season" in df.columns else "N/A",
        "year_range": (
            f"{int(df['Crop_Year'].min())} – {int(df['Crop_Year'].max())}"
            if "Crop_Year" in df.columns else "N/A"
        ),
    }


# ─────────────────────────────────────────────
# 3. Clean data
# ─────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Steps:
      - Strip whitespace from string columns
      - Drop rows where Area or Production is zero / null
      - Compute Yield = Production / Area  (kg/hectare target variable)
      - Remove extreme outliers (IQR method on Yield)
    """
    df = df.copy()

    # Strip strings
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())

    # Drop nulls in critical columns
    df.dropna(subset=["Area", "Production"], inplace=True)

    # Remove zeros to avoid division errors
    df = df[(df["Area"] > 0) & (df["Production"] > 0)]

    # Create target variable
    df["Yield"] = df["Production"] / df["Area"]

    # Remove outliers using IQR on Yield
    Q1 = df["Yield"].quantile(0.01)
    Q3 = df["Yield"].quantile(0.99)
    df = df[(df["Yield"] >= Q1) & (df["Yield"] <= Q3)]

    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────
# 4. Feature engineering
# ─────────────────────────────────────────────
def encode_features(df: pd.DataFrame):
    """
    Label-encode categorical features.
    Returns:
      - df_encoded  : DataFrame with encoded columns
      - encoders    : dict of {col: LabelEncoder} for inverse transform
    """
    df = df.copy()
    cat_cols = ["State_Name", "District_Name", "Season", "Crop"]
    encoders = {}

    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    return df, encoders


def get_feature_matrix(df_encoded: pd.DataFrame):
    """
    Return X (features) and y (target) ready for ML.
    Features used: State, District, Season, Crop (encoded) + Crop_Year + Area
    """
    feature_cols = [
        c for c in [
            "State_Name_enc", "District_Name_enc", "Season_enc",
            "Crop_enc", "Crop_Year", "Area"
        ] if c in df_encoded.columns
    ]
    X = df_encoded[feature_cols]
    y = df_encoded["Yield"]
    return X, y, feature_cols
