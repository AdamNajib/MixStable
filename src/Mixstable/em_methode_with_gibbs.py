import numpy as np
from sklearn.cluster import KMeans
from .utils import *
from .ecf_estimators import *
from .gibbs import mock_gibbs_sampling

def em_estimate_stable_recursive_ecf_with_gibbs(data, max_iter=100, tol=1e-4):
    """
    EM algorithm using recursive ECF for alpha-stable mixture with Gibbs-based M-step.
    
    Returns:
        params1, params2: dictionaries with alpha, beta, gamma, delta
        w: weight of first component
    """

    data = np.asarray(data)
    u = np.linspace(0.1, 1, 10)

    # === Initialization via clustering ===
    kmeans = KMeans(n_clusters=2, random_state=42).fit(data.reshape(-1, 1))
    labels = kmeans.labels_
    w = np.mean(labels == 0)

    if np.sum(labels == 0) < 5:
        params1 = dict(zip(['alpha', 'beta', 'gamma', 'delta'], stable_fit_init(data)))
    else:
        params1 = estimate_stable_recursive_ecf(data[labels == 0], u)

    if np.sum(labels == 1) < 5:
        params2 = dict(zip(['alpha', 'beta', 'gamma', 'delta'], stable_fit_init(data)))
    else:
        params2 = estimate_stable_recursive_ecf(data[labels == 1], u)

    log_likelihood_old = -np.inf

    for iteration in range(max_iter):
        # E-step
        pdf1 = np.maximum(r_stable_pdf(data, *unpack_params(params1)), 1e-300)
        pdf2 = np.maximum(r_stable_pdf(data, *unpack_params(params2)), 1e-300)
        resp1 = w * pdf1
        resp2 = (1 - w) * pdf2
        total = resp1 + resp2
        gamma1 = resp1 / total
        gamma2 = resp2 / total

        # M-step with Gibbs
        labels = (gamma1 > gamma2).astype(int)
        w = np.mean(labels == 0)

        if np.sum(labels == 0) >= 5:
            best1, _ = mock_gibbs_sampling(data[labels == 0])
            params1 = dict(zip(['alpha', 'beta', 'gamma', 'delta'], best1[1:5]))
        else:
            print("⚠️ Cluster 0 too small. Reusing previous estimate.")

        if np.sum(labels == 1) >= 5:
            best2, _ = mock_gibbs_sampling(data[labels == 1])
            params2 = dict(zip(['alpha', 'beta', 'gamma', 'delta'], best2[5:9]))
        else:
            print("⚠️ Cluster 1 too small. Reusing previous estimate.")

        # Log-likelihood
        pdf1 = np.maximum(r_stable_pdf(data, *unpack_params(params1)), 1e-300)
        pdf2 = np.maximum(r_stable_pdf(data, *unpack_params(params2)), 1e-300)
        log_likelihood = np.sum(np.log(w * pdf1 + (1 - w) * pdf2))

        print(f"[Gibbs EM] Iteration {iteration}: Log-Likelihood = {log_likelihood:.6f}")

        if abs(log_likelihood - log_likelihood_old) < tol:
            print(f"✅ Converged after {iteration} iterations.")
            break

        log_likelihood_old = log_likelihood

    return params1, params2, w


def em_estimate_stable_kernel_ecf_with_gibbs(data, max_iter=100, tol=1e-4):
    """
    EM algorithm with Gibbs-based M-step to fit a mixture of two alpha-stable distributions
    using the kernel-based ECF method for initialization and likelihood tracking.
    """
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
        pdf1 = np.maximum(r_stable_pdf(data, *unpack_params(params1)), 1e-300)
        pdf2 = np.maximum(r_stable_pdf(data, *unpack_params(params2)), 1e-300)

        resp1 = w * pdf1
        resp2 = (1 - w) * pdf2
        total = resp1 + resp2
        gamma1 = resp1 / total
        gamma2 = resp2 / total

        labels = (gamma1 > gamma2).astype(int)
        w = np.mean(labels == 0)

        # Gibbs-enhanced M-step
        if np.sum(labels == 0) >= 5:
            best1, _ = mock_gibbs_sampling(data[labels == 0])
            params1 = dict(zip(['alpha', 'beta', 'gamma', 'delta'], best1[1:5]))
        else:
            print("⚠️ Cluster 0 too small. Reusing previous estimate.")

        if np.sum(labels == 1) >= 5:
            best2, _ = mock_gibbs_sampling(data[labels == 1])
            params2 = dict(zip(['alpha', 'beta', 'gamma', 'delta'], best2[5:9]))
        else:
            print("⚠️ Cluster 1 too small. Reusing previous estimate.")

        # Log-likelihood
        pdf1 = np.maximum(r_stable_pdf(data, *unpack_params(params1)), 1e-300)
        pdf2 = np.maximum(r_stable_pdf(data, *unpack_params(params2)), 1e-300)
        log_likelihood = np.sum(np.log(w * pdf1 + (1 - w) * pdf2))

        print(f"[Gibbs EM] Iteration {iteration}: Log-Likelihood = {log_likelihood:.6f}")

        if abs(log_likelihood - log_likelihood_old) < tol:
            print(f"✅ Converged after {iteration} iterations.")
            break

        log_likelihood_old = log_likelihood

    return params1, params2, w

def em_estimate_stable_weighted_ols_with_gibbs(data, max_iter=100, tol=1e-4):
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
        pdf1 = np.maximum(r_stable_pdf(data, *unpack_params(params1)), 1e-300)
        pdf2 = np.maximum(r_stable_pdf(data, *unpack_params(params2)), 1e-300)

        resp1 = w * pdf1
        resp2 = (1 - w) * pdf2
        total = resp1 + resp2
        gamma1 = resp1 / total
        gamma2 = resp2 / total

        labels = (gamma1 > gamma2).astype(int)
        w = np.mean(labels == 0)

        # Gibbs-based M-step
        if np.sum(labels == 0) >= 5:
            best1, _ = mock_gibbs_sampling(data[labels == 0])
            params1 = dict(zip(['alpha', 'beta', 'gamma', 'delta'], best1[1:5]))
        else:
            print("⚠️ Cluster 0 too small. Reusing previous estimate.")

        if np.sum(labels == 1) >= 5:
            best2, _ = mock_gibbs_sampling(data[labels == 1])
            params2 = dict(zip(['alpha', 'beta', 'gamma', 'delta'], best2[5:9]))
        else:
            print("⚠️ Cluster 1 too small. Reusing previous estimate.")

        pdf1 = np.maximum(r_stable_pdf(data, *unpack_params(params1)), 1e-300)
        pdf2 = np.maximum(r_stable_pdf(data, *unpack_params(params2)), 1e-300)
        log_likelihood = np.sum(np.log(w * pdf1 + (1 - w) * pdf2))

        print(f"[Gibbs EM] Iteration {iteration}: Log-Likelihood = {log_likelihood:.6f}")

        if abs(log_likelihood - log_likelihood_old) < tol:
            print(f"✅ Converged after {iteration} iterations.")
            break

        log_likelihood_old = log_likelihood

    return params1, params2, w

def em_estimate_stable_from_cdf_with_gibbs(data, max_iter=100, tol=1e-4):
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
        pdf1 = np.maximum(r_stable_pdf(data, *unpack_params(params1)), 1e-300)
        pdf2 = np.maximum(r_stable_pdf(data, *unpack_params(params2)), 1e-300)

        resp1 = w * pdf1
        resp2 = (1 - w) * pdf2
        total = resp1 + resp2
        gamma1 = resp1 / total
        gamma2 = resp2 / total

        labels = (gamma1 > gamma2).astype(int)
        w = np.mean(labels == 0)

        # Gibbs-based M-step
        if np.sum(labels == 0) >= 5:
            best1, _ = mock_gibbs_sampling(data[labels == 0])
            params1 = dict(zip(['alpha', 'beta', 'gamma', 'delta'], best1[1:5]))
        else:
            print("⚠️ Cluster 0 too small. Reusing previous estimate.")

        if np.sum(labels == 1) >= 5:
            best2, _ = mock_gibbs_sampling(data[labels == 1])
            params2 = dict(zip(['alpha', 'beta', 'gamma', 'delta'], best2[5:9]))
        else:
            print("⚠️ Cluster 1 too small. Reusing previous estimate.")

        pdf1 = np.maximum(r_stable_pdf(data, *unpack_params(params1)), 1e-300)
        pdf2 = np.maximum(r_stable_pdf(data, *unpack_params(params2)), 1e-300)
        log_likelihood = np.sum(np.log(w * pdf1 + (1 - w) * pdf2))

        print(f"[Gibbs EM] Iteration {iteration}: Log-Likelihood = {log_likelihood:.6f}")

        if abs(log_likelihood - log_likelihood_old) < tol:
            print(f"✅ Converged after {iteration} iterations.")
            break

        log_likelihood_old = log_likelihood

    return params1, params2, w

