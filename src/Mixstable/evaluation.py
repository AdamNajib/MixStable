from .mcculloch import mcculloch_lookup_estimate, build_mcculloch_interpolators, generate_mcculloch_table
from .ecf import ecf_estimate_all
from .mle import mle_estimate
import numpy as np
from scipy.stats import levy_stable
from .generate_sample import generate_mixture_data
from .em import em_fit_alpha_stable_mixture
from .em_methode_with_gibbs import em_estimate_stable_kernel_ecf_with_gibbs
from .utils import r_stable_pdf
import pandas as pd

# Step 1: Generate the lookup table
alpha_grid = np.linspace(0.6, 2.0, 15)
beta_grid = np.linspace(-1.0, 1.0, 15)
table = generate_mcculloch_table(alpha_grid, beta_grid)

# Step 2: Build the interpolators
interp_alpha, interp_beta = build_mcculloch_interpolators(table)


def evaluate_estimation_method(estimator_fn, true_params, n=1000, trials=20, seed=42):
    """
    Run estimator_fn on samples from true_params and compute MSE.

    Parameters:
        estimator_fn : callable returning dict with keys α, β, γ, δ
        true_params : dict with keys α, β, γ, δ
        n : sample size
        trials : number of simulations

    Returns:
        mean MSE across trials
    """
    np.random.seed(seed)
    mse_list = []

    for _ in range(trials):
        X = levy_stable.rvs(true_params["alpha"], true_params["beta"],
                            loc=true_params["delta"], scale=true_params["gamma"], size=n)
        est = estimator_fn(X)
        mse = np.mean([(est[k] - true_params[k])**2 for k in ['alpha', 'beta', 'gamma', 'delta']])
        mse_list.append(mse)

    return np.mean(mse_list)

def compare_methods_across_configs(parameter_configs, trials=20, n=1000):
    results = {}
    for name, true_params in parameter_configs.items():
        print(f"Running: {name}")
        
        # McCulloch
        mc_mse = evaluate_estimation_method(
            lambda X: mcculloch_lookup_estimate(X, interp_alpha, interp_beta),
            true_params, n=n, trials=trials
        )

        # ECF
        ecf_mse = evaluate_estimation_method(
            lambda X: ecf_estimate_all(X),
            true_params, n=n, trials=trials
        )

        # MLE
        mle_mse = evaluate_estimation_method(
            lambda X: mle_estimate(X),
            true_params, n=n, trials=trials
        )

        results[name] = {
            "McCulloch": mc_mse,
            "ECF": ecf_mse,
            "MLE": mle_mse
        }
    return results


def compare_em_vs_em_gibbs(data,n_runs=20):
    results = []
    for seed in range(n_runs):
        # Standard EM
        params_em1, params_em2, w_em= em_fit_alpha_stable_mixture(data)
        alpha1, beta1, gamma1, delta1 = params_em1
        alpha2, beta2, gamma2, delta2 = params_em2
        pdf_em = w_em * r_stable_pdf(data, alpha1,beta1, gamma1, delta1) + (1 - w_em) * r_stable_pdf(data, alpha2, beta2, gamma2, delta2)
        loglik_em = np.sum(np.log(np.clip(pdf_em, 1e-300, None)))

        # EM with Gibbs (kernel ECF)
        params_gibbs1, params_gibbs2, w_gibbs = em_estimate_stable_kernel_ecf_with_gibbs(data, n_runs, tol=1e-4)
        alpha_gibbs1, beta_gibbs1, gamma_gibbs1, delta_gibbs1 = params_gibbs1
        alpha_gibbs2, beta_gibbs2, gamma_gibbs2, delta_gibbs2 = params_gibbs2
        pdf_gibbs = w_gibbs * r_stable_pdf(data, alpha_gibbs1,beta_gibbs1, gamma_gibbs1, delta_gibbs1) + (1 - w_gibbs) * r_stable_pdf(data, alpha_gibbs2, beta_gibbs2, gamma_gibbs2, delta_gibbs2)
        loglik_gibbs = np.sum(np.log(np.clip(pdf_gibbs, 1e-300, None)))

        results.append({
            "seed": seed,
            "loglik_em": loglik_em,
            "loglik_gibbs": loglik_gibbs,
            "w_em": w_em,
            "w_gibbs": w_gibbs,
            "params_em1": params_em1,
            "params_em2": params_em2,
            "params_gibbs1": params_gibbs1,
            "params_gibbs2": params_gibbs2
        })

    df = pd.DataFrame(results)
    return df