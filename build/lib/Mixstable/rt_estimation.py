# rt_estimation.py
import numpy as np
from scipy.optimize import minimize

def RT(inc, G):
    """
    Compute the effective reproduction number Rt over time.

    Parameters:
    - inc: array-like, incidence time series
    - G: array-like, generation time distribution

    Returns:
    - Rt: array of Rt values
    """
    m = len(inc)
    n = len(G)
    R = np.zeros(m)
    for i in range(1, m):
        t = 0
        for j in range(n):
            if i != j:
                t += inc[abs(i - j)] * G[j]
        R[i] = inc[i] / t if t > 0 else 0
    return R

def est_r0_ml(W, N):
    """
    Estimate R0 using maximum likelihood from incidence and generation time.

    Parameters:
    - W: array-like, generation time distribution
    - N: array-like, incidence time series

    Returns:
    - R0: estimated basic reproduction number
    """
    m = len(N)
    v = np.zeros(m)
    for i in range(1, m):
        t = 0
        for j in range(min(i + 1, len(W))):
            if 0 <= i - j < len(N):
                t += N[i - j] * W[j]
        v[i] = t
    R0 = np.sum(N[1:]) / np.sum(v[1:]) if np.sum(v[1:]) > 0 else 0
    return R0

def empirical_r0(incidence, serial_interval, growth_model="exponential"):
    """
    Estimate R0 using empirical growth rate method.

    Parameters:
    - incidence: array-like, incidence time series
    - serial_interval: float, average serial interval
    - growth_model: str, "exponential" or "logistic"

    Returns:
    - R0 estimate
    """
    if len(incidence) < 2:
        raise ValueError("Incidence data must contain at least two points.")
    if serial_interval <= 0:
        raise ValueError("Serial interval must be greater than zero.")

    incidence = np.array(incidence)
    try:
        if growth_model == "exponential":
            growth_rate = np.polyfit(np.arange(len(incidence)), np.log(incidence + 1e-10), 1)[0]
        elif growth_model == "logistic":
            growth_rate = np.polyfit(np.arange(len(incidence)), np.log(incidence / (1 + incidence) + 1e-10), 1)[0]
        else:
            raise ValueError(f"Unsupported growth model: {growth_model}")
        return 1 + growth_rate * serial_interval
    except Exception as e:
        raise RuntimeError(f"Error in empirical R0 estimation: {e}")

def est_r0_mle(incidence, gen_time):
    """
    Estimate R0 using maximum likelihood with Poisson likelihood.

    Parameters:
    - incidence: array-like, incidence time series
    - gen_time: float, generation time

    Returns:
    - R0 estimate
    """
    if len(incidence) < 2:
        raise ValueError("Incidence data must contain at least two points.")
    if gen_time <= 0:
        raise ValueError("Generation time must be greater than zero.")

    def neg_log_likelihood(r0):
        ll = 0
        for t in range(1, len(incidence)):
            lambda_t = r0 * np.sum(incidence[:t] * np.exp(-np.arange(t, 0, -1) / gen_time))
            ll += -lambda_t + incidence[t] * np.log(lambda_t + 1e-10)
        return -ll

    try:
        result = minimize(neg_log_likelihood, x0=[2.0], bounds=[(0.1, 5)])
        if not result.success:
            raise RuntimeError(f"MLE optimization failed: {result.message}")
        return result.x[0]
    except Exception as e:
        raise RuntimeError(f"Error in MLE R0 estimation: {e}")
