import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data.reshape(-1, 1)).flatten()

def extract_serial_intervals(df):
    """
    Extracts serial intervals from a DataFrame.
    Priority:
    1. Use 'serial_interval' column if it exists.
    2. Otherwise, compute it from 'infector_onsetDate' and 'infectee_onsetDate'.
    """
    if "serial_interval" in df.columns:
        return df["serial_interval"].astype(float).dropna().values

    # Try to compute from onset dates
    required_cols = ["infector_onsetDate", "infectee_onsetDate"]
    if all(col in df.columns for col in required_cols):
        df["infector_onsetDate"] = pd.to_datetime(df["infector_onsetDate"], dayfirst=True, errors='coerce')
        df["infectee_onsetDate"] = pd.to_datetime(df["infectee_onsetDate"], dayfirst=True, errors='coerce')
        df["SI"] = (df["infectee_onsetDate"] - df["infector_onsetDate"]).dt.days
        return df["SI"].dropna().astype(float).values

    raise ValueError("No valid 'serial_interval' column or onset date columns found.")
