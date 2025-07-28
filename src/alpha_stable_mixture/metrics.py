import numpy as np
from .utils import r_stable_pdf

def calculate_log_likelihood(data, params):
    return -np.sum((data - np.mean(data))**2)

def aic(log_likelihood, num_params):
    return 2 * num_params - 2 * log_likelihood

def bic(log_likelihood, num_params, n):
    return np.log(n) * num_params - 2 * log_likelihood

def compute_model_metrics(data, params):
    try:
        pdf_vals = r_stable_pdf(data, *params)
        log_likelihood = np.sum(np.log(np.clip(pdf_vals, 1e-300, None)))
        k = len(params)
        n = len(data)
        aic = -2 * log_likelihood + 2 * k
        bic = -2 * log_likelihood + k * np.log(n)
        return {
            "log_likelihood": log_likelihood,
            "AIC": aic,
            "BIC": bic
        }
    except Exception as e:
        return {
            "log_likelihood": np.nan,
            "AIC": np.nan,
            "BIC": np.nan,
            "error": str(e)
        }
