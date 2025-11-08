import numpy as np

def compute_quantile_ratios(X):
    """
    Compute quantile-based ratios v_alpha, v_beta, v_gamma, v_delta.

    Returns:
        dict with ratios
    """
    x05 = np.quantile(X, 0.05)
    x25 = np.quantile(X, 0.25)
    x50 = np.quantile(X, 0.5)
    x75 = np.quantile(X, 0.75)
    x95 = np.quantile(X, 0.95)

    v_alpha = (x95 - x05) / (x75 - x25)
    v_beta = (x05 + x95 - 2 * x50) / (x95 - x05)
    v_gamma = x75 - x25
    v_delta = -x50

    return {
        "v_alpha": v_alpha,
        "v_beta": v_beta,
        "v_gamma": v_gamma,
        "v_delta": v_delta
    }