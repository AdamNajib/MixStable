import numpy as np
from sklearn.linear_model import LinearRegression
from .utils import ecf_fn, eta0, stable_fit_init
import statsmodels.api as sm
from sklearn.linear_model import HuberRegressor

def CDF(x, u):
    _, _, gamma0, delta0 = stable_fit_init(x)
    x = (x - delta0) / gamma0

    y1, y3 = ecf_fn(x, u)
    z1 = np.log(-np.log(y1))
    v1 = np.log(np.abs(u))
    a1 = np.polyfit(v1, z1, 1)
    alpha1 = np.clip(a1[0], 0.1, 2)
    gamma1 = np.exp(a1[1] / alpha1)

    v3 = -gamma1**alpha1 * eta0(u, alpha1, gamma1)
    a3 = np.linalg.lstsq(np.column_stack([v3, u]), y3, rcond=None)[0]
    beta1 = np.clip(a3[0], -1, 1)
    delta1 = a3[1]

    return {
        "alpha": alpha1,
        "beta": beta1,
        "gamma": gamma1 * gamma0,
        "delta": delta1 * gamma0 + delta0
    }

def ecf_regression(x, u):
    """
    Estimate stable distribution parameters using ECF regression.

    Parameters:
    - x: input data array
    - u: frequency values (u)

    Returns:
    - dict with estimated parameters: alpha, beta, gamma, delta
    """

    # Step 1: Normalize data using initial gamma and delta
    gamma0 = stable_fit_init(x)[2]
    delta0 = stable_fit_init(x)[3]
    x = (np.array(x) - delta0) / gamma0

    # Step 2: Compute ECF
    yr, yi = ecf_fn(x, u, method='kernel')
    y1 = yr
    z1 = np.log(-np.log(y1))
    v1 = np.log(np.abs(u))

    # Step 3: First regression (log-log)
    reg1 = LinearRegression().fit(v1.reshape(-1, 1), z1)
    residuals1 = np.abs(z1 - reg1.predict(v1.reshape(-1, 1)))
    # Avoid division by zero in weights
    pred1 = reg1.predict(v1.reshape(-1, 1)).reshape(-1, 1)
    fit_resid1 = LinearRegression().fit(pred1, residuals1)
    fit_pred1 = fit_resid1.predict(pred1)
    fit_pred1[fit_pred1 == 0] = 1e-8
    weights1 = 1 / (fit_pred1 ** 2)
    reg1.fit(v1.reshape(-1, 1), z1, sample_weight=weights1.flatten())

    alpha1 = min(max(reg1.coef_[0], 0.1), 2)
    gamma1 = np.exp(reg1.intercept_ / alpha1)

    # Step 4: Second regression (imaginary part)
    y3 = yi
    v3 = -gamma1 ** alpha1 * eta0(u, alpha1, gamma1)

    Xmat = np.column_stack((v3, u))
    reg3 = LinearRegression(fit_intercept=False)
    reg3.fit(Xmat, y3)
    residuals3 = np.abs(y3 - reg3.predict(Xmat))
    pred3 = reg3.predict(Xmat).reshape(-1, 1)
    fit_resid3 = LinearRegression().fit(pred3, residuals3)
    fit_pred3 = fit_resid3.predict(pred3)
    fit_pred3[fit_pred3 == 0] = 1e-8
    weights3 = 1 / (fit_pred3 ** 2)

    reg3.fit(Xmat, y3, sample_weight=weights3.flatten())
    beta1, delta1 = reg3.coef_

    beta1 = min(max(beta1, -1), 1)

    return {
        "alpha": alpha1,
        "beta": beta1,
        "gamma": gamma1 * gamma0,
        "delta": delta1 * gamma0 + delta0
    }

def robust_ecf_regression(x, u):
    gamma0 = stable_fit_init(x)[2]
    delta0 = stable_fit_init(x)[3]
    x = (np.array(x) - delta0) / gamma0

    yr, yi = ecf_fn(x, u, method='kernel')
    z1 = np.log(-np.log(np.clip(yr, 1e-10, 1)))
    v1 = np.log(np.abs(u))

    huber1 = HuberRegressor().fit(v1.reshape(-1, 1), z1)
    alpha1 = np.clip(huber1.coef_[0], 0.1, 2)
    gamma1 = np.exp(huber1.intercept_ / alpha1)

    v3 = -gamma1 ** alpha1 * eta0(u, alpha1, gamma1)
    Xmat = np.column_stack((v3, u))

    huber2 = HuberRegressor(fit_intercept=False).fit(Xmat, yi)
    beta1, delta1 = huber2.coef_
    beta1 = np.clip(beta1, -1, 1)

    return {
        "alpha": alpha1,
        "beta": beta1,
        "gamma": gamma1 * gamma0,
        "delta": delta1 * gamma0 + delta0
    }

def fit_stable_ecf(data, frequencies):
    """
    Fit stable distribution parameters using the empirical characteristic function (ECF)
    with kernel smoothing and weighted regression.

    Parameters:
    - data: array-like, input data
    - frequencies: array-like, frequency values (u)

    Returns:
    - dict: estimated parameters {alpha, beta, gamma, delta}
    """
    # Step 1: Normalize data
    _, _, gamma0, delta0 = stable_fit_init(data)
    normalized_data = (data - delta0) / gamma0

    # Step 2: Compute ECF
    ecf_real, ecf_imag = ecf_fn(normalized_data, frequencies, method='kernel')
    log_modulus = np.log(-np.log(np.maximum(ecf_real, 1e-10)))
    log_freq = np.log(np.abs(frequencies))

    # Step 3: First regression (log-log)
    reg1 = LinearRegression().fit(log_freq.reshape(-1, 1), log_modulus)
    residuals1 = np.abs(log_modulus - reg1.predict(log_freq.reshape(-1, 1)))
    weights1 = 1 / (LinearRegression().fit(
        reg1.predict(log_freq.reshape(-1, 1)).reshape(-1, 1),
        residuals1
    ).predict(reg1.predict(log_freq.reshape(-1, 1)).reshape(-1, 1)) ** 2)
    reg1.fit(log_freq.reshape(-1, 1), log_modulus, sample_weight=weights1)

    alpha = np.clip(reg1.coef_[0], 0.1, 2)
    gamma = np.exp(reg1.intercept_ / alpha)

    # Step 4: Second regression (imaginary part)
    eta_vals = -gamma ** alpha * eta0(frequencies, alpha, gamma)
    X = np.column_stack((eta_vals, frequencies))

    # Filter invalid values
    valid = np.all(np.isfinite(X), axis=1) & np.isfinite(ecf_imag)
    if not np.any(valid):
        raise ValueError("Filtered arrays are empty. Check input data.")

    X = X[valid]
    y3 = ecf_imag[valid]

    reg2 = LinearRegression(fit_intercept=False).fit(X, y3)
    residuals2 = np.abs(y3 - reg2.predict(X))
    weights2 = 1 / (LinearRegression().fit(
        reg2.predict(X).reshape(-1, 1),
        residuals2
    ).predict(reg2.predict(X).reshape(-1, 1)) ** 2)
    reg2.fit(X, y3, sample_weight=weights2)

    beta, delta = reg2.coef_
    beta = np.clip(beta, -1, 1)

    return {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma * gamma0,
        "delta": delta * gamma0 + delta0
    }

def estimate_stable_recursive_ecf(x, u): #CDF
    """Estimate parameters using recursive ECF."""
    _, _, gamma0, delta0 = stable_fit_init(x)
    x = (x - delta0) / gamma0
    x1 = ecf_fn(x, u, method='recursive')
    y1 = np.clip(x1[0], 1e-10, 1)
    z1 = np.log(-np.log(y1))
    v1 = np.log(np.abs(u))
    reg1 = LinearRegression().fit(v1.reshape(-1, 1), z1)
    alpha1 = np.clip(reg1.coef_[0], 0.1, 2)
    gamma1 = np.exp(reg1.intercept_ / alpha1)
    y3 = x1[1]
    v3 = -gamma1 ** alpha1 * eta0(u, alpha1, gamma1)
    X = np.column_stack((v3, u))
    valid_indices = np.all(np.isfinite(X), axis=1) & np.isfinite(y3)
    X = X[valid_indices]
    y3 = y3[valid_indices]
    
    # Check if the filtered arrays are empty
    if X.size == 0 or y3.size == 0:
        print("Warning: Filtered arrays are empty. Returning default parameter values.")
        return {"alpha": alpha1, "beta": 0, "gamma": gamma1 * gamma0, "delta": delta0}
    
    reg3 = LinearRegression(fit_intercept=False).fit(X, y3)
    beta1, delta1 = reg3.coef_
    beta1 = np.clip(beta1, -1, 1)
    return {"alpha": alpha1, "beta": beta1, "gamma": gamma1 * gamma0, "delta": delta1 * gamma0 + delta0}

def estimate_stable_kernel_ecf(x, u): # CDF1
    """Estimate parameters using kernel ECF."""
    _, _, gamma0, delta0 = stable_fit_init(x)
    x = (x - delta0) / gamma0
    y1, y3 = ecf_fn(x, u, method='kernel')
    y1 = np.clip(y1, 1e-10, 1)
    z1 = np.log(-np.log(y1))
    v1 = np.log(np.abs(u))
    valid_indices = np.isfinite(z1)
    z1 = z1[valid_indices]
    v1 = v1[valid_indices]
    reg1 = LinearRegression().fit(v1.reshape(-1, 1), z1)
    alpha1 = np.clip(reg1.coef_[0], 0.1, 2)
    gamma1 = np.exp(reg1.intercept_ / alpha1)
    v3 = -gamma1 ** alpha1 * eta0(u, alpha1, gamma1)
    X = np.column_stack((v3, u))
    valid_indices = np.all(np.isfinite(X), axis=1) & np.isfinite(y3)
    X = X[valid_indices]
    y3 = y3[valid_indices]
    reg3 = LinearRegression(fit_intercept=False).fit(X, y3)
    beta1, delta1 = reg3.coef_
    beta1 = np.clip(beta1, -1, 1)
    return {"alpha": alpha1, "beta": beta1, "gamma": gamma1 * gamma0, "delta": delta1 * gamma0 + delta0}

def estimate_stable_weighted_ols(x, u): # CDF2 (i need to put h_n = b_n instead of h_n = SJ)
    """Estimate parameters using weighted OLS."""
    _, _, gamma0, delta0 = stable_fit_init(x)
    x = (x - delta0) / gamma0
    x1 = ecf_fn(x, u, method='recursive')
    y1 = np.clip(x1[0], 1e-10, 1)
    z1 = np.log(-np.log(y1))
    v1 = np.log(np.abs(u))
    X1 = sm.add_constant(v1)
    model1 = sm.OLS(z1, X1).fit()
    alpha1 = np.clip(model1.params[1], 0.1, 2)
    gamma1 = np.exp(model1.params[0] / alpha1)
    y3 = x1[1]
    v3 = -gamma1 ** alpha1 * eta0(u, alpha1, gamma1)
    X3 = np.column_stack([v3, u])
    valid_indices = np.all(np.isfinite(X3), axis=1) & np.isfinite(y3)
    X3 = X3[valid_indices]
    y3 = y3[valid_indices]
    
    # Check if the filtered arrays are empty
    if X3.size == 0 or y3.size == 0:
        print("Warning: Filtered arrays are empty. Returning default parameter values.")
        return {'alpha': alpha1, 'beta': 0, 'gamma': gamma1 * gamma0, 'delta': delta0}
    
    # Perform weighted least squares regression
    model2 = sm.WLS(y3, X3).fit()
    weights = 1 / np.maximum(np.abs(model2.resid), 1e-8) ** 2  
    model2 = sm.WLS(y3, X3, weights=weights).fit()
    beta1 = np.clip(model2.params[0], -1, 1)
    delta1 = model2.params[1]
    return {'alpha': alpha1, 'beta': beta1, 'gamma': gamma1 * gamma0, 'delta': delta1 * gamma0 + delta0}

def estimate_stable_from_cdf(x, u): # CDF3
    """Estimate parameters using CDF."""
    _, _, gamma0, delta0 = stable_fit_init(x)
    x = (x - delta0) / gamma0
    y1, _ = ecf_fn(x, u)
    y1 = np.clip(y1, 1e-10, 1)
    z1 = np.log(-np.log(y1))
    v1 = np.log(np.abs(u))
    X1 = sm.add_constant(v1)
    model1 = sm.OLS(z1, X1).fit()
    alpha1 = np.clip(model1.params[1], 0.1, 2)
    gamma1 = np.exp(model1.params[0] / alpha1)
    _, y3 = ecf_fn(x, u)
    v3 = -gamma1 ** alpha1 * eta0(u, alpha1, gamma1)
    X3 = np.column_stack([v3, u])
    model2 = sm.WLS(y3, X3).fit()
    weights = 1 / np.maximum(np.abs(model2.resid), 1e-8) ** 2
    model2 = sm.WLS(y3, X3, weights=weights).fit()
    beta1 = np.clip(model2.params[0], -1, 1)
    delta1 = model2.params[1]
    return {'alpha': alpha1, 'beta': beta1, 'gamma': gamma1 * gamma0, 'delta': delta1 * gamma0 + delta0}

def ecf_empirical(X, t_grid):
    """
    Empirical Characteristic Function estimator φ̂ₙ(t)

    Parameters:
        X : array-like, shape (n,)
            Data sample
        t_grid : array-like, shape (m,)
            Grid of t values at which to evaluate the characteristic function

    Returns:
        phi_vals : complex ndarray, shape (m,)
            Estimated characteristic function values at each t
    """
    X = np.asarray(X)
    phi_vals = np.array([np.mean(np.exp(1j * t * X)) for t in t_grid])
    return phi_vals

def ecf_components(phi_vals):
    """
    Compute log(-log|φ̂|) and arg(φ̂) for regression

    Returns:
        y_vals : array-like
            log(-log(|φ̂(t)|)) values
        arg_vals : array-like
            Arguments (angles) of φ̂(t)
    """
    abs_vals = np.abs(phi_vals)
    y_vals = np.log(-np.log(abs_vals))
    arg_vals = np.angle(phi_vals)
    return y_vals, arg_vals

