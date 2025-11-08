import numpy as np
from sklearn.cluster import KMeans
from .utils import r_stable_pdf
from .mle import log_likelihood_mixture
import scipy.stats

def mock_gibbs_sampling(data, n_samples=500, verbose=False):
    """
    Improved Gibbs-style sampler for alpha-stable mixture parameters using a likelihood-based selection.

    Parameters:
        data (array-like): 1D array of observed data.
        n_samples (int): Number of Gibbs samples to draw.
        verbose (bool): Whether to print the best log-likelihood.

    Returns:
        tuple: (best_params, samples)
            - best_params: list of 9 parameters [w, α1, β1, γ1, δ1, α2, β2, γ2, δ2]
            - samples: list of (log_likelihood, parameter_list) tuples
    """

    samples = []
    best_loglik = -np.inf
    best_params = None

    data_mean = np.mean(data)
    data_std = np.std(data)

    for _ in range(n_samples):
        try:
            # Component 1 priors (heavier tail)
            alpha1 = np.random.uniform(0.8, 1.6)               # wider α range for tails
            beta1 = np.random.uniform(-1.0, 1.0)
            gamma1 = max(np.abs(np.random.normal(1.0, 0.5)), 0.1)  # minimum scale to avoid spikiness
            delta1 = np.random.normal(data_mean, data_std)

            # Component 2 priors (lighter tail)
            alpha2 = np.random.uniform(1.2, 2.0)
            beta2 = np.random.uniform(-1.0, 1.0)
            gamma2 = max(np.abs(np.random.normal(1.5, 0.5)), 0.1)
            delta2 = np.random.normal(data_mean + 2.0, data_std)

            # Mixture weight prior (balanced preference)
            w = np.clip(np.random.beta(2, 2), 0.05, 0.95)

            candidate = [w, alpha1, beta1, gamma1, delta1, alpha2, beta2, gamma2, delta2]
            ll = -log_likelihood_mixture(candidate, data)

            samples.append((ll, candidate))

            if ll > best_loglik:
                best_loglik = ll
                best_params = candidate

        except Exception:
            continue  # skip invalid parameter sets silently

    if verbose:
        print(f"✅ Best Gibbs+MLE Log-Likelihood: {best_loglik:.2f}")

    return best_params, samples

def gibbs_sampler(data, iterations=1000):
    """
    Gibbs sampling for 2-component Gaussian mixture.
    """
    if len(data) < 2:
        raise ValueError("Data must contain at least two points.")

    n = len(data)
    z = np.random.randint(0, 2, size=n)
    mu = [np.mean(data) - 1, np.mean(data) + 1]
    sigma = [1.0, 1.0]
    pi = 0.5

    samples = []

    for it in range(iterations):
        # Step 1: Sample z
        for i in range(n):
            p0 = pi * scipy.stats.norm.pdf(data[i], mu[0], sigma[0])
            p1 = (1 - pi) * scipy.stats.norm.pdf(data[i], mu[1], sigma[1])
            total_prob = p0 + p1
            if total_prob > 0:
                z[i] = np.random.choice([0, 1], p=[p0 / total_prob, p1 / total_prob])
            else:
                z[i] = np.random.choice([0, 1])  # Randomly assign if probabilities are invalid

        # Step 2: Sample mu, sigma
        for j in [0, 1]:
            group = data[z == j]
            n_j = len(group)
            mu[j] = np.mean(group) if n_j > 0 else mu[j]
            sigma[j] = np.std(group) if n_j > 0 else sigma[j]

        # Step 3: Sample pi
        pi = np.mean(z == 0)

        samples.append((mu[0], mu[1], sigma[0], sigma[1], pi))

    return samples

# MCMC Sampling with Metropolis-Hastings
def metropolis_hastings(fct,iterations, lok, aa=[1, 1], proposal_std=0.1):
    """
    Metropolis-Hastings algorithm for parameter estimation in a mixture of stable distributions.

    Parameters:
        iterations (int): Number of iterations.
        lok (array-like): Input data.
        aa (list): Dirichlet prior for weights.
        proposal_std (float): Standard deviation for proposal distribution.

    Returns:
        tuple: Updated weights and parameters for both components.
    """
    np.random.seed(123)
    n = len(lok)
    u1 = np.linspace(0.1, 1.0, 10)

    # Initial clustering
    a = KMeans(n_clusters=2, random_state=123).fit_predict(lok.reshape(-1, 1))
    lok1 = fct(lok[a == 0], u1)
    lok2 = fct(lok[a == 1], u1)

    # Initialize parameters
    M2_alpha1 = [lok1["alpha"]]
    M2_beta1 = [lok1["beta"]]
    M2_delta1 = [lok1["delta"]]
    M2_omega1 = [lok1["gamma"]]

    M2_alpha2 = [lok2["alpha"]]
    M2_beta2 = [lok2["beta"]]
    M2_delta2 = [lok2["delta"]]
    M2_omega2 = [lok2["gamma"]]

    M2_w1 = [np.mean(a == 0)]
    M2_w2 = [np.mean(a == 1)]

    M2_cc = np.zeros((iterations + 1, n), dtype=int)
    M2_cc[0] = a + 1  # Adjusting cluster labels to be 1 and 2

    for s in range(iterations):
        cc = M2_cc[s, :]

        # Update weights
        counts = np.array([np.sum(cc == 1), np.sum(cc == 2)])
        w = np.random.dirichlet(aa + counts)
        w1, w2 = w
        M2_w1.append(w1)
        M2_w2.append(w2)

        # Propose new parameters
        def propose(param, lower, upper):
            proposal = np.random.normal(param, 0.1)
            while proposal < lower or proposal > upper:
                proposal = np.random.normal(param, 0.1)
            return proposal

        alpha1_star = propose(M2_alpha1[s], 0.1, 2)
        beta1_star = propose(M2_beta1[s], -1, 1)
        delta1_star = propose(M2_delta1[s], 1e-3, np.inf)
        omega1_star = propose(M2_omega1[s], 1e-3, np.inf)

        alpha2_star = propose(M2_alpha2[s], 0.1, 2)
        beta2_star = propose(M2_beta2[s], -1, 1)
        delta2_star = propose(M2_delta2[s], 1e-3, np.inf)
        omega2_star = propose(M2_omega2[s], 1e-3, np.inf)

        p1_star = [alpha1_star, beta1_star, delta1_star, omega1_star]
        p2_star = [alpha2_star, beta2_star, delta2_star, omega2_star]

        # Compute log densities
        lok_array = np.array(lok)
        idx1 = cc == 1
        idx2 = cc == 2
        
        p1 = [M2_alpha1[s], M2_beta1[s], M2_delta1[s], M2_omega1[s]]
        p2 = [M2_alpha2[s], M2_beta2[s], M2_delta2[s], M2_omega2[s]]

        log_pdf1_current = np.log(r_stable_pdf(lok_array[idx1], *p1) + 1e-10)
        log_pdf1_proposed = np.log(r_stable_pdf(lok_array[idx1], *p1_star) + 1e-10)

        log_pdf2_current = np.log(r_stable_pdf(lok_array[idx2], *p2) + 1e-10)
        log_pdf2_proposed = np.log(r_stable_pdf(lok_array[idx2], *p2_star) + 1e-10)

        # Acceptance probabilities
        accept1 = np.exp(np.sum(log_pdf1_proposed) - np.sum(log_pdf1_current))
        accept2 = np.exp(np.sum(log_pdf2_proposed) - np.sum(log_pdf2_current))

        # Accept or reject proposals
        if np.random.rand() < accept1:
            M2_alpha1.append(alpha1_star)
            M2_beta1.append(beta1_star)
            M2_delta1.append(delta1_star)
            M2_omega1.append(omega1_star)
        else:
            M2_alpha1.append(M2_alpha1[s])
            M2_beta1.append(M2_beta1[s])
            M2_delta1.append(M2_delta1[s])
            M2_omega1.append(M2_omega1[s])

        if np.random.rand() < accept2:
            M2_alpha2.append(alpha2_star)
            M2_beta2.append(beta2_star)
            M2_delta2.append(delta2_star)
            M2_omega2.append(omega2_star)
        else:
            M2_alpha2.append(M2_alpha2[s])
            M2_beta2.append(M2_beta2[s])
            M2_delta2.append(M2_delta2[s])
            M2_omega2.append(M2_omega2[s])

        # Update cluster assignments

        params1 = [M2_alpha1[-1], M2_beta1[-1], M2_delta1[-1], M2_omega1[-1]]
        params2 = [M2_alpha2[-1], M2_beta2[-1], M2_delta2[-1], M2_omega2[-1]]

        log_pdf1 = np.log(r_stable_pdf(lok_array, *params1) + 1e-10)
        log_pdf2 = np.log(r_stable_pdf(lok_array, *params2) + 1e-10)

        log_prob1 = np.log(w1 + 1e-10) + log_pdf1
        log_prob2 = np.log(w2 + 1e-10) + log_pdf2

        max_log = np.maximum(log_prob1, log_prob2)
        prob1 = np.exp(log_prob1 - max_log)
        prob2 = np.exp(log_prob2 - max_log)
        sum_probs = prob1 + prob2
        prob1 /= sum_probs
        prob2 /= sum_probs

        new_cc = np.random.binomial(1, prob2) + 1
        M2_cc[s + 1, :] = new_cc

        print(f"Iteration {s + 1}/{iterations} completed.")

    return M2_w1, M2_w2, M2_alpha1, M2_beta1, M2_delta1, M2_omega1, M2_alpha2, M2_beta2, M2_delta2, M2_omega2