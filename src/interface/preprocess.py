import pandas as pd
import io

def read_csv_with_auto_delimiter(uploaded_file):
    try:
        # Try reading with default comma delimiter
        df = pd.read_csv(uploaded_file)
    except Exception:
        # Try reading with semicolon if comma fails
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, delimiter=';')
    return df

def extract_serial_intervals(df):
    possible_cols = ["serial_interval", "serial_interval_mean_based", "mean_serial_interval"]
    serial_col = next((col for col in possible_cols if col in df.columns), None)
    if serial_col:
        return df[serial_col].astype(float).dropna().values
    else:
        numeric_cols = df.select_dtypes(include=[float, int]).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric column found in the uploaded file.")
        return df[numeric_cols[0]].astype(float).dropna().values

