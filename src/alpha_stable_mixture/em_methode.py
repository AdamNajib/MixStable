import numpy as np
from sklearn.cluster import KMeans
from .utils import r_stable_pdf,stable_fit_init
from .ecf_estimators import *
from numpy import maximum

def unpack_params(p):
    """
    Helper to unpack parameter dictionary into tuple.
    """
    return p['alpha'], p['beta'], p['gamma'], p['delta']

def em_estimate_stable_recursive_ecf(data, max_iter=100, tol=1e-4):
    """
    EM algorithm using recursive ECF for alpha-stable mixture.

    Returns:
        params1, params2: dictionaries with alpha, beta, gamma, delta
        w: weight of first component
    """

    data = np.asarray(data)
    n = len(data)
    u = np.linspace(0.1, 1, 10)  # use local u

    # === Initialization via clustering ===
    kmeans = KMeans(n_clusters=2, random_state=42).fit(data.reshape(-1, 1))
    labels = kmeans.labels_
    w = np.mean(labels == 0)

    if np.sum(labels == 0) < 5:
        params1 = dict(zip(['alpha','beta','gamma','delta'], stable_fit_init(data)))
    else:
        params1 = estimate_stable_recursive_ecf(data[labels == 0], u)

    if np.sum(labels == 1) < 5:
        params2 = dict(zip(['alpha','beta','gamma','delta'], stable_fit_init(data)))
    else:
        params2 = estimate_stable_recursive_ecf(data[labels == 1], u)

    log_likelihood_old = -np.inf

    for iteration in range(max_iter):
        # E-step
        pdf1 = maximum(r_stable_pdf(data, *unpack_params(params1)), 1e-300)
        pdf2 = maximum(r_stable_pdf(data, *unpack_params(params2)), 1e-300)
        resp1 = w * pdf1
        resp2 = (1 - w) * pdf2
        total = resp1 + resp2
        gamma1 = resp1 / total
        gamma2 = resp2 / total

        # M-step
        labels = (gamma1 > gamma2).astype(int)
        w = np.mean(labels == 0)

        # Avoid collapsing clusters
        if np.sum(labels == 0) >= 5:
            params1 = estimate_stable_recursive_ecf(data[labels == 0], u)
        else:
            print("Cluster 0 too small. Reusing previous estimate.")

        if np.sum(labels == 1) >= 5:
            params2 = estimate_stable_recursive_ecf(data[labels == 1], u)
        else:
            print("Cluster 1 too small. Reusing previous estimate.")

        # Log-likelihood
        pdf1 = maximum(r_stable_pdf(data, *unpack_params(params1)), 1e-300)
        pdf2 = maximum(r_stable_pdf(data, *unpack_params(params2)), 1e-300)
        log_likelihood = np.sum(np.log(w * pdf1 + (1 - w) * pdf2))

        if abs(log_likelihood - log_likelihood_old) < tol:
            print(f"Converged after {iteration} iterations.")
            break
        log_likelihood_old = log_likelihood

    return params1, params2, w

def em_estimate_stable_kernel_ecf(data, max_iter=100, tol=1e-4):
    data = np.asarray(data)
    u = np.linspace(0.1, 1, 10)
    kmeans = KMeans(n_clusters=2, random_state=42).fit(data.reshape(-1, 1))
    labels = kmeans.labels_
    w = np.mean(labels == 0)

    if np.sum(labels == 0) < 5:
        params1 = dict(zip(['alpha', 'beta', 'gamma', 'delta'], stable_fit_init(data)))
    else:
        params1 = estimate_stable_kernel_ecf(data[labels == 0], u)

    if np.sum(labels == 1) < 5:
        params2 = dict(zip(['alpha', 'beta', 'gamma', 'delta'], stable_fit_init(data)))
    else:
        params2 = estimate_stable_kernel_ecf(data[labels == 1], u)

    log_likelihood_old = -np.inf

    for iteration in range(max_iter):
        pdf1 = maximum(r_stable_pdf(data, *unpack_params(params1)), 1e-300)
        pdf2 = maximum(r_stable_pdf(data, *unpack_params(params2)), 1e-300)
        resp1 = w * pdf1
        resp2 = (1 - w) * pdf2
        total = resp1 + resp2
        gamma1 = resp1 / total
        gamma2 = resp2 / total

        labels = (gamma1 > gamma2).astype(int)
        w = np.mean(labels == 0)

        if np.sum(labels == 0) >= 5:
            params1 = estimate_stable_kernel_ecf(data[labels == 0], u)
        else:
            print("Cluster 0 too small. Reusing previous estimate.")

        if np.sum(labels == 1) >= 5:
            params2 = estimate_stable_kernel_ecf(data[labels == 1], u)
        else:
            print("Cluster 1 too small. Reusing previous estimate.")

        pdf1 = maximum(r_stable_pdf(data, *unpack_params(params1)), 1e-300)
        pdf2 = maximum(r_stable_pdf(data, *unpack_params(params2)), 1e-300)
        log_likelihood = np.sum(np.log(w * pdf1 + (1 - w) * pdf2))

        if abs(log_likelihood - log_likelihood_old) < tol:
            print(f"Converged after {iteration} iterations.")
            break
        log_likelihood_old = log_likelihood

    return params1, params2, w

def em_estimate_stable_weighted_ols(data, max_iter=100, tol=1e-4):
    data = np.asarray(data)
    u = np.linspace(0.1, 1, 10)
    kmeans = KMeans(n_clusters=2, random_state=42).fit(data.reshape(-1, 1))
    labels = kmeans.labels_
    w = np.mean(labels == 0)

    if np.sum(labels == 0) < 5:
        params1 = dict(zip(['alpha', 'beta', 'gamma', 'delta'], stable_fit_init(data)))
    else:
        params1 = estimate_stable_weighted_ols(data[labels == 0], u)

    if np.sum(labels == 1) < 5:
        params2 = dict(zip(['alpha', 'beta', 'gamma', 'delta'], stable_fit_init(data)))
    else:
        params2 = estimate_stable_weighted_ols(data[labels == 1], u)

    log_likelihood_old = -np.inf

    for iteration in range(max_iter):
        pdf1 = maximum(r_stable_pdf(data, *unpack_params(params1)), 1e-300)
        pdf2 = maximum(r_stable_pdf(data, *unpack_params(params2)), 1e-300)
        resp1 = w * pdf1
        resp2 = (1 - w) * pdf2
        total = resp1 + resp2
        gamma1 = resp1 / total
        gamma2 = resp2 / total

        labels = (gamma1 > gamma2).astype(int)
        w = np.mean(labels == 0)

        if np.sum(labels == 0) >= 5:
            params1 = estimate_stable_weighted_ols(data[labels == 0], u)
        else:
            print("Cluster 0 too small. Reusing previous estimate.")

        if np.sum(labels == 1) >= 5:
            params2 = estimate_stable_weighted_ols(data[labels == 1], u)
        else:
            print("Cluster 1 too small. Reusing previous estimate.")

        pdf1 = maximum(r_stable_pdf(data, *unpack_params(params1)), 1e-300)
        pdf2 = maximum(r_stable_pdf(data, *unpack_params(params2)), 1e-300)
        log_likelihood = np.sum(np.log(w * pdf1 + (1 - w) * pdf2))

        if abs(log_likelihood - log_likelihood_old) < tol:
            print(f"Converged after {iteration} iterations.")
            break
        log_likelihood_old = log_likelihood

    return params1, params2, w

def em_estimate_stable_from_cdf(data, max_iter=100, tol=1e-4):
    data = np.asarray(data)
    u = np.linspace(0.1, 1, 10)
    kmeans = KMeans(n_clusters=2, random_state=42).fit(data.reshape(-1, 1))
    labels = kmeans.labels_
    w = np.mean(labels == 0)

    if np.sum(labels == 0) < 5:
        params1 = dict(zip(['alpha', 'beta', 'gamma', 'delta'], stable_fit_init(data)))
    else:
        params1 = estimate_stable_from_cdf(data[labels == 0], u)

    if np.sum(labels == 1) < 5:
        params2 = dict(zip(['alpha', 'beta', 'gamma', 'delta'], stable_fit_init(data)))
    else:
        params2 = estimate_stable_from_cdf(data[labels == 1], u)

    log_likelihood_old = -np.inf

    for iteration in range(max_iter):
        pdf1 = maximum(r_stable_pdf(data, *unpack_params(params1)), 1e-300)
        pdf2 = maximum(r_stable_pdf(data, *unpack_params(params2)), 1e-300)
        resp1 = w * pdf1
        resp2 = (1 - w) * pdf2
        total = resp1 + resp2
        gamma1 = resp1 / total
        gamma2 = resp2 / total

        labels = (gamma1 > gamma2).astype(int)
        w = np.mean(labels == 0)

        if np.sum(labels == 0) >= 5:
            params1 = estimate_stable_from_cdf(data[labels == 0], u)
        else:
            print("Cluster 0 too small. Reusing previous estimate.")

        if np.sum(labels == 1) >= 5:
            params2 = estimate_stable_from_cdf(data[labels == 1], u)
        else:
            print("Cluster 1 too small. Reusing previous estimate.")

        pdf1 = maximum(r_stable_pdf(data, *unpack_params(params1)), 1e-300)
        pdf2 = maximum(r_stable_pdf(data, *unpack_params(params2)), 1e-300)
        log_likelihood = np.sum(np.log(w * pdf1 + (1 - w) * pdf2))

        if abs(log_likelihood - log_likelihood_old) < tol:
            print(f"Converged after {iteration} iterations.")
            break
        log_likelihood_old = log_likelihood

    return params1, params2, w

