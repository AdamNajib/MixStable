import streamlit as st
import pandas as pd
import numpy as np
from alpha_stable_mixture.em import em_fit_alpha_stable_mixture
from alpha_stable_mixture.visualization import plot_final_mixture_fit
from alpha_stable_mixture.ecf_estimators import (
    estimate_stable_kernel_ecf,
    estimate_stable_weighted_ols,
    estimate_stable_from_cdf
)
from alpha_stable_mixture.mle import fit_alpha_stable_mle
from alpha_stable_mixture.metrics import compute_model_metrics
from alpha_stable_mixture.utils import wasserstein_distance_mixture
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Mixstable", layout="wide")
st.title("Mixstable")

# Data source selection
data_source = st.radio(
    "Select data source:",
    ("Upload CSV file", "Use test data")
)

uploaded_file = None
df = None
test_data_folder = "Data"
test_files = [f for f in os.listdir(test_data_folder) if f.endswith('.csv')]
selected_test_file = None

if data_source == "Upload CSV file":
    uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])
elif data_source == "Use test data":
    if test_files:
        selected_test_file = st.selectbox("Choose a test CSV file:", test_files)
        if selected_test_file:
            st.info(f"Using test data from '{selected_test_file}'")
            try:
                df = pd.read_csv(os.path.join(test_data_folder, selected_test_file), sep=",", decimal=".")
            except Exception as e:
                st.error(f"Could not load test data: {e}")
    else:
        st.warning("No test CSV files found in the Data folder.")

use_discrete_bins = st.checkbox("☑ Discrete Bins")

methods = {
    "ECF - Kernel": estimate_stable_kernel_ecf,
    "ECF - Recursive": estimate_stable_weighted_ols,
    "MLE": fit_alpha_stable_mle,
    "ECF - CDF": estimate_stable_from_cdf
}
selected_method = st.selectbox("Choose estimation method", list(methods.keys()))
estimator_func = methods[selected_method]

run_estimation = False
if data_source == "Upload CSV file":
    run_estimation = uploaded_file is not None and st.button("Run Estimation")
elif data_source == "Use test data":
    run_estimation = df is not None and st.button("Run Estimation")

if run_estimation:
    try:
        if data_source == "Upload CSV file":
            df = pd.read_csv(uploaded_file, sep=",", decimal=".")

        # Extract data column
        if "serial_interval" in df.columns:
            data = df["serial_interval"].astype(float).dropna().values
        else:
            # Try to infer the correct column if not known
            numeric_cols = df.select_dtypes(include=[float, int]).columns
            if len(numeric_cols) == 0:
                st.error("No numeric column found in the CSV file. Please specify the correct column name.")
                st.stop()
            data = df[numeric_cols[0]].astype(float).dropna().values

        st.success("✅ Data loaded and processed.")

        u = np.linspace(0.1, 1, 20)
        n_components = 2  # Fixed for simplicity; can be made dynamic
        with st.spinner("🧠 Running EM algorithm..."):
            result = em_fit_alpha_stable_mixture(data, max_iter=200, tol=1e-4)

        st.success("🎯 EM estimation complete.")

        st.subheader("Estimated Parameters")
        params_data = {
            "alpha": [result['params'][i]['alpha'] for i in range(n_components)],
            "beta": [result['params'][i]['beta'] for i in range(n_components)],
            "gamma": [result['params'][i]['gamma'] for i in range(n_components)],
            "delta": [result['params'][i]['delta'] for i in range(n_components)]
        }
        st.table(pd.DataFrame(params_data))
        # --- Metrics Table ---
        st.subheader("Model Metrics")
        try:
            w_distance = wasserstein_distance_mixture(data, result['params'], result['weights'])
            metrics = compute_model_metrics(data, result['params'], result['weights'])
            metrics_table = {
                "Metric": ["Wasserstein Distance"] + list(metrics.keys()),
                "Value": [w_distance] + list(metrics.values())
            }
            st.table(pd.DataFrame(metrics_table))
        except Exception as e:
            st.warning(f"Could not compute metrics: {e}")

        st.subheader("Distribution Fit")
        plot_final_mixture_fit(data, *result['params'], result['weights'])
        st.image("mixture_alpha_stable_fit_final.png")

    except Exception as e:
        st.error(f"❌ Failed to process file: {e}")
        st.warning("Please ensure the file contains 'serial_interval' or valid numeric columns.")
else:
    st.info("📂 Please select a data source, upload a CSV file or use test data, and click 'Run Estimation' to begin.")