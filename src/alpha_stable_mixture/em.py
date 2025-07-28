import numpy as np
from sklearn.cluster import KMeans
from .utils import ensure_positive_scale
from .utils import r_stable_pdf,stable_fit_init
import numpy as np
from sklearn.cluster import KMeans
from .mle import fit_alpha_stable_mle
from scipy.optimize import minimize
import scipy
from .ecf import ecf_estimate_all
from .ecf_estimators import estimate_stable_weighted_ols

def simple_em_real(X, max_iter=10):
    """
    Simple 2-component EM using ECF initialization on real dataset.
    """
    from sklearn.cluster import KMeans
    clusters = KMeans(n_clusters=2).fit(X.reshape(-1, 1)).labels_
    lambda1 = np.mean(clusters == 0)

    ests = {}
    for k in [0, 1]:
        Xi = X[clusters == k]
        ests[k] = ecf_estimate_all(Xi)

    return {
        "lambda1": lambda1,
        "params1": ests[0],
        "params2": ests[1]
    }

# === EM Algorithm for alpha-stable mixtures ===
def em_alpha_stable(data, n_components=2, max_iter=100, tol=1e-4, random_init=True, debug=True):
    """
    EM algorithm for fitting a mixture of alpha-stable distributions.

    Parameters:
    - data: array-like, input data
    - n_components: int, number of mixture components
    - max_iter: int, maximum iterations
    - tol: float, convergence tolerance
    - random_init: bool, whether to use random initialization
    - debug: bool, print debug info

    Returns:
    - dict of estimated parameters
    """
    N = len(data)
    if random_init:
        weights = np.ones(n_components) / n_components
        alphas = np.random.uniform(1.2, 1.8, n_components)
        betas = np.random.uniform(-0.5, 0.5, n_components)
        gammas = np.random.uniform(0.5, 2.0, n_components)
        deltas = np.random.uniform(np.min(data), np.max(data), n_components)
    else:
        weights = np.ones(n_components) / n_components
        alphas = np.full(n_components, 1.8)
        betas = np.zeros(n_components)
        gammas = np.full(n_components, np.std(data) / 2)
        deltas = np.linspace(np.min(data), np.max(data), n_components)

    log_likelihood_old = -np.inf

    for iteration in range(max_iter):
        responsibilities = np.zeros((N, n_components))
        for k in range(n_components):
            parms = (alphas[k], betas[k], gammas[k], deltas[k])
            pdf_vals = r_stable_pdf(data, *parms)
            pdf_vals = np.clip(pdf_vals, 1e-300, None)
            responsibilities[:, k] = weights[k] * pdf_vals

        sum_responsibilities = responsibilities.sum(axis=1, keepdims=True) + 1e-12
        responsibilities /= sum_responsibilities

        for k in range(n_components):
            r = responsibilities[:, k]
            Nk = np.sum(r)
            if Nk < 1e-8:
                continue
            weights[k] = Nk / N
            expanded_data = np.repeat(data, np.round(r / r.sum() * N).astype(int))
            if len(expanded_data) > 10 and np.std(expanded_data) > 1e-8:
                try:
                    params = stable_fit_init(expanded_data)
                    alphas[k], betas[k], gammas[k], deltas[k] = params
                except Exception as e:
                    if debug:
                        print(f"Fit failed for component {k}: {e}")

        likelihood = np.zeros((N, n_components))
        for k in range(n_components):
            parms_k = (alphas[k], betas[k], gammas[k], deltas[k])
            likelihood[:, k] = weights[k] * r_stable_pdf(data, *parms_k)

        total_likelihood = np.sum(np.log(np.sum(likelihood, axis=1) + 1e-12))
        if debug:
            print(f"[Iteration {iteration + 1}] Log-Likelihood: {total_likelihood:.6f}")
        if np.abs(total_likelihood - log_likelihood_old) < tol:
            if debug:
                print(f"Converged after {iteration + 1} iterations.")
            break
        log_likelihood_old = total_likelihood

    return {
        'weights': weights,
        'alphas': alphas,
        'betas': betas,
        'gammas': gammas,
        'deltas': deltas
    }

def em_stable_mixture(data, u, estimator_func, max_iter=300, epsilon=1e-3):
    np.random.seed(134)
    S = data
    n = len(S)

    # Initial clustering
    kmeans = KMeans(n_clusters=2, random_state=134).fit(S.reshape(-1, 1))
    labels = kmeans.labels_

    # Initial parameter estimation
    S1 = estimator_func(S[labels == 0], u)
    S2 = estimator_func(S[labels == 1], u)

    w = np.mean(labels == 0)
    p1 = [S1['alpha'], S1['beta'], ensure_positive_scale(S1['delta']), ensure_positive_scale(S1['gamma'])]
    p2 = [S2['alpha'], S2['beta'], ensure_positive_scale(S2['delta']), ensure_positive_scale(S2['gamma'])]

    LV = -np.inf
    for s in range(max_iter):
        cc = np.zeros(n, dtype=int)

        for i in range(n):
            try:
                v1 = np.log(w) + np.log(r_stable_pdf(S[i:i+1], *p1)[0] + 1e-10)
                v2 = np.log(1 - w) + np.log(r_stable_pdf(S[i:i+1], *p2)[0] + 1e-10)
                v = np.exp([v1, v2] - np.max([v1, v2]))
                v = v / np.sum(v) if np.sum(v) > 0 else np.array([0.5, 0.5])
                v = np.clip(v, 0, 1)
            except Exception:
                v = np.array([0.5, 0.5])

            cc[i] = np.random.choice([0, 1], p=v)

        w = np.clip(np.mean(cc == 0), 0.01, 0.99)

        if np.sum(cc == 0) >= 2:
            try:
                L1 = estimator_func(S[cc == 0], u)
                if all(np.isfinite([L1[k] for k in ['alpha', 'beta', 'delta', 'gamma']])):
                    p1 = [L1['alpha'], L1['beta'], ensure_positive_scale(L1['delta']), ensure_positive_scale(L1['gamma'])]
            except Exception:
                pass

        if np.sum(cc == 1) >= 2:
            try:
                L2 = estimator_func(S[cc == 1], u)
                if all(np.isfinite([L2[k] for k in ['alpha', 'beta', 'delta', 'gamma']])):
                    p2 = [L2['alpha'], L2['beta'], ensure_positive_scale(L2['delta']), ensure_positive_scale(L2['gamma'])]
            except Exception:
                pass

        LVn = np.sum(np.log(w * r_stable_pdf(S, *p1) + (1 - w) * r_stable_pdf(S, *p2)))
        if abs(LVn - LV) / abs(LVn) < epsilon:
            break
        LV = LVn
        print(f"Iteration {s+1}, Log-likelihood: {LVn}")

    return {
        "weights": w,
        "params1": p1,
        "params2": p2,
        "log_likelihood": LV
    }


# 🔁 EM algorithm
def em_fit_alpha_stable_mixture(data, max_iter=200, tol=1e-4):
    """
    EM algorithm to fit a mixture of two alpha-stable distributions.

    Parameters:
        data (array-like): Input data.
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance.

    Returns:
        tuple: Parameters of the two components and the mixture weight.
    """
    if len(data) < 2:
        raise ValueError("Input data must contain at least two points.")

    # Initialize clusters using KMeans
    kmeans = KMeans(n_clusters=2, random_state=134).fit(data.reshape(-1, 1))
    labels = kmeans.labels_

    # Initial parameter estimation
    params1 = fit_alpha_stable_mle(data[labels == 0])
    params2 = fit_alpha_stable_mle(data[labels == 1])
    w = np.mean(labels == 0)

    log_likelihood = -np.inf

    for iteration in range(max_iter):
        # E-step: Compute responsibilities
        pdf1 = np.maximum(r_stable_pdf(data, *params1), 1e-300)
        pdf2 = np.maximum(r_stable_pdf(data, *params2), 1e-300)
        responsibilities = np.vstack([w * pdf1, (1 - w) * pdf2]).T
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)

        # M-step: Update parameters
        labels = np.argmax(responsibilities, axis=1)
        w = np.mean(labels == 0)

        if np.sum(labels == 0) >= 2:
            params1 = fit_alpha_stable_mle(data[labels == 0])
        if np.sum(labels == 1) >= 2:
            params2 = fit_alpha_stable_mle(data[labels == 1])

        # Compute log-likelihood
        total_pdf = w * pdf1 + (1 - w) * pdf2
        new_log_likelihood = np.sum(np.log(total_pdf))

        print(f"Iteration {iteration}: Log-Likelihood = {new_log_likelihood:.6f}")

        # Check for convergence
        if abs(new_log_likelihood - log_likelihood) / abs(new_log_likelihood) < tol:
            print("Converged.")
            break

        log_likelihood = new_log_likelihood

    return params1, params2, w

def em_estimation_mixture(data, max_iter=100, tol=1e-6):
    """
    EM algorithm for a Gaussian mixture (2 components).
    """
    if len(data) < 2:
        raise ValueError("Data must contain at least two points.")

    n = len(data)
    pi = 0.5
    mu1, mu2 = np.min(data), np.max(data)
    sigma1, sigma2 = 1.0, 1.0

    for i in range(max_iter):
        # E-step
        resp1 = pi * scipy.stats.norm.pdf(data, mu1, sigma1)
        resp2 = (1 - pi) * scipy.stats.norm.pdf(data, mu2, sigma2)
        sum_resp = resp1 + resp2
        w1 = resp1 / sum_resp
        w2 = resp2 / sum_resp

        # M-step
        pi_new = np.mean(w1)
        mu1_new = np.sum(w1 * data) / np.sum(w1)
        mu2_new = np.sum(w2 * data) / np.sum(w2)
        sigma1_new = np.sqrt(np.sum(w1 * (data - mu1_new)**2) / np.sum(w1))
        sigma2_new = np.sqrt(np.sum(w2 * (data - mu2_new)**2) / np.sum(w2))

        # Convergence check
        if np.abs(mu1 - mu1_new) < tol and np.abs(mu2 - mu2_new) < tol:
            break

        pi, mu1, mu2, sigma1, sigma2 = pi_new, mu1_new, mu2_new, sigma1_new, sigma2_new

    return {
        'pi': pi, 'mu1': mu1, 'sigma1': sigma1,
        'mu2': mu2, 'sigma2': sigma2
    }



