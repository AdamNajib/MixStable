import numpy as np
from .em_methode import *
from .em_methode_with_gibbs import *
from .visualization import plot_mixture_fit

def run_all_estimations(X1, bw_sj, max_iter=200, tol=1e-4):
    """
    Run all stable mixture model estimation methods and plot the results.

    Parameters:
    - X1: np.ndarray, the data
    - bw_sj: float, bandwidth selector value (from R or precomputed)
    - max_iter: int, maximum number of EM iterations
    - tol: float, tolerance for convergence
    """

    X1 = X1[~np.isnan(X1)]
    h_n = bw_sj
    print(f"Bande passante R (bw.SJ) : {bw_sj:.5f}")

    # Recursive ECF Estimation
    try:
        params1, params2, w = em_estimate_stable_recursive_ecf(X1, max_iter=max_iter, tol=tol)
    except ValueError as e:
        print(f"em_estimate_stable_recursive_ecf failed: {e}")
        params1 = params2 = {'alpha': np.nan, 'beta': np.nan, 'gamma': np.nan, 'delta': np.nan}
        w = 0.5

    # Kernel ECF Estimation
    try:
        params11, params22, w1 = em_estimate_stable_kernel_ecf(X1, max_iter=max_iter, tol=tol)
    except ValueError as e:
        print(f"em_estimate_stable_kernel_ecf failed: {e}")
        params11 = params22 = {'alpha': np.nan, 'beta': np.nan, 'gamma': np.nan, 'delta': np.nan}
        w1 = 0.5

    # Weighted OLS Estimation
    try:
        params111, params222, w2 = em_estimate_stable_weighted_ols(X1, max_iter=max_iter, tol=tol)
    except ValueError as e:
        print(f"em_estimate_stable_weighted_ols failed: {e}")
        params111 = params222 = {'alpha': np.nan, 'beta': np.nan, 'gamma': np.nan, 'delta': np.nan}
        w2 = 0.5

    # CDF-based Estimation
    try:
        params1111, params2222, w3 = em_estimate_stable_from_cdf(X1, max_iter=max_iter, tol=tol)
    except ValueError as e:
        print(f"em_estimate_stable_from_cdf failed: {e}")
        params1111 = params2222 = {'alpha': np.nan, 'beta': np.nan, 'gamma': np.nan, 'delta': np.nan}
        w3 = 0.5

    # Organize estimated parameters
    estimated_params = {
        'weights': [w, 1-w],
        'alphas': [params1['alpha'], params2['alpha']],
        'betas': [params1['beta'], params2['beta']],
        'gammas': [params1['gamma'], params2['gamma']],
        'deltas': [params1['delta'], params2['delta']],
    }

    estimated_params1 = {
        'weights': [w1, 1-w1],
        'alphas': [params11['alpha'], params22['alpha']],
        'betas': [params11['beta'], params22['beta']],
        'gammas': [params11['gamma'], params22['gamma']],
        'deltas': [params11['delta'], params22['delta']],
    }

    estimated_params2 = {
        'weights': [w2, 1-w2],
        'alphas': [params111['alpha'], params222['alpha']],
        'betas': [params111['beta'], params222['beta']],
        'gammas': [params111['gamma'], params222['gamma']],
        'deltas': [params111['delta'], params222['delta']],
    }

    estimated_params3 = {
        'weights': [w3, 1-w3],
        'alphas': [params1111['alpha'], params2222['alpha']],
        'betas': [params1111['beta'], params2222['beta']],
        'gammas': [params1111['gamma'], params2222['gamma']],
        'deltas': [params1111['delta'], params2222['delta']],
    }

    # Plot all results
    plot_mixture_fit(X1, estimated_params)
    plot_mixture_fit(X1, estimated_params1)
    plot_mixture_fit(X1, estimated_params2)
    plot_mixture_fit(X1, estimated_params3)

import numpy as np

def run_enzyme_estimations_with_gibbs(enzyme_data, bw_sj, max_iter=100, tol=1e-4):
    """
    Run stable mixture model estimation methods with Gibbs sampling on enzyme data.

    Parameters:
    - enzyme_data: R data frame column or list (converted to NumPy array)
    - bw_sj: float, bandwidth selector
    - max_iter: int, max EM iterations
    - tol: float, convergence tolerance
    """

    X1 = np.array(enzyme_data)
    X1 = X1[~np.isnan(X1)]

    print(f"Bande passante R (bw.SJ) : {bw_sj:.5f}")

    # Recursive ECF with Gibbs
    try:
        params1, params2, w = em_estimate_stable_recursive_ecf_with_gibbs(X1, max_iter=max_iter, tol=tol)
    except ValueError as e:
        print(f"em_estimate_stable_recursive_ecf failed: {e}")
        params1 = params2 = {'alpha': np.nan, 'beta': np.nan, 'gamma': np.nan, 'delta': np.nan}
        w = 0.5

    # Kernel ECF with Gibbs
    try:
        params11, params22, w1 = em_estimate_stable_kernel_ecf_with_gibbs(X1, max_iter=max_iter, tol=tol)
    except ValueError as e:
        print(f"em_estimate_stable_kernel_ecf failed: {e}")
        params11 = params22 = {'alpha': np.nan, 'beta': np.nan, 'gamma': np.nan, 'delta': np.nan}
        w1 = 0.5

    # Weighted OLS with Gibbs
    try:
        params111, params222, w2 = em_estimate_stable_weighted_ols_with_gibbs(X1, max_iter=max_iter, tol=tol)
    except ValueError as e:
        print(f"em_estimate_stable_weighted_ols failed: {e}")
        params111 = params222 = {'alpha': np.nan, 'beta': np.nan, 'gamma': np.nan, 'delta': np.nan}
        w2 = 0.5

    # CDF with Gibbs
    try:
        params1111, params2222, w3 = em_estimate_stable_from_cdf_with_gibbs(X1, max_iter=max_iter, tol=tol)
    except ValueError as e:
        print(f"em_estimate_stable_from_cdf failed: {e}")
        params1111 = params2222 = {'alpha': np.nan, 'beta': np.nan, 'gamma': np.nan, 'delta': np.nan}
        w3 = 0.5

    # Assemble estimated parameters
    estimated_params = {
        'weights': [w, 1-w],
        'alphas': [params1['alpha'], params2['alpha']],
        'betas': [params1['beta'], params2['beta']],
        'gammas': [params1['gamma'], params2['gamma']],
        'deltas': [params1['delta'], params2['delta']],
    }

    estimated_params1 = {
        'weights': [w1, 1-w1],
        'alphas': [params11['alpha'], params22['alpha']],
        'betas': [params11['beta'], params22['beta']],
        'gammas': [params11['gamma'], params22['gamma']],
        'deltas': [params11['delta'], params22['delta']],
    }

    estimated_params2 = {
        'weights': [w2, 1-w2],
        'alphas': [params111['alpha'], params222['alpha']],
        'betas': [params111['beta'], params222['beta']],
        'gammas': [params111['gamma'], params222['gamma']],
        'deltas': [params111['delta'], params222['delta']],
    }

    estimated_params3 = {
        'weights': [w3, 1-w3],
        'alphas': [params1111['alpha'], params2222['alpha']],
        'betas': [params1111['beta'], params2222['beta']],
        'gammas': [params1111['gamma'], params2222['gamma']],
        'deltas': [params1111['delta'], params2222['delta']],
    }

    # Plot results
    plot_mixture_fit(X1, estimated_params)
    plot_mixture_fit(X1, estimated_params1)
    plot_mixture_fit(X1, estimated_params2)
    plot_mixture_fit(X1, estimated_params3)
