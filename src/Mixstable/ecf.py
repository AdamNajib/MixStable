import numpy as np
from sklearn.linear_model import LinearRegression
from .utils import eta_func
from .ecf_estimators import ecf_empirical

def estimate_alpha_gamma(t_grid, phi_vals, weights=None):
    """
    Estimate alpha and gamma via linear regression on log-log ECF magnitude.

    Parameters:
        t_grid : array-like
            Grid of t values
        phi_vals : complex ndarray
            ECF values at each t
        weights : optional, array-like
            Observation weights for regression

    Returns:
        alpha_hat, gamma_hat
    """
    y = np.log(-np.log(np.abs(phi_vals)))
    X = np.log(np.abs(t_grid)).reshape(-1, 1)
    
    if weights is not None:
        reg = LinearRegression()
        reg.fit(X, y, sample_weight=weights)
    else:
        reg = LinearRegression().fit(X, y)
    
    alpha_hat = reg.coef_[0]
    a_hat = reg.intercept_
    gamma_hat = np.exp(a_hat / alpha_hat)
    
    return alpha_hat, gamma_hat

def estimate_beta_delta(t_grid, phi_vals, alpha_hat, gamma_hat, weights=None):
    """
    Estimate beta and delta via regression on arg(φ̂).

    Parameters:
        t_grid : array-like
        phi_vals : complex ndarray
        alpha_hat, gamma_hat : estimated parameters

    Returns:
        beta_hat, delta_hat
    """
    z_k = np.angle(phi_vals)
    eta_k = eta_func(t_grid, alpha_hat, gamma_hat)
    B_k = gamma_hat ** alpha_hat * eta_k
    A = np.vstack([B_k, t_grid]).T

    if weights is not None:
        reg = LinearRegression()
        reg.fit(A, z_k, sample_weight=weights)
    else:
        reg = LinearRegression().fit(A, z_k)
    
    beta_hat = reg.coef_[0]
    delta_hat = reg.coef_[1]
    return beta_hat, delta_hat

def ecf_estimate_all(X, t_grid=None, weights=None):
    """
    Estimate (α, β, γ, δ) from sample X using ECF regression.

    Returns:
        dict with keys: alpha, beta, gamma, delta
    """
    if t_grid is None:
        t_grid = np.linspace(0.1, 1.0, 50)
    
    phi_vals = ecf_empirical(X, t_grid)
    alpha_hat, gamma_hat = estimate_alpha_gamma(t_grid, phi_vals, weights)
    beta_hat, delta_hat = estimate_beta_delta(t_grid, phi_vals, alpha_hat, gamma_hat, weights)

    return {
        'alpha': alpha_hat,
        'beta': beta_hat,
        'gamma': gamma_hat,
        'delta': delta_hat
    }