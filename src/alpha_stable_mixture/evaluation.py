from .mcculloch import mcculloch_lookup_estimate, interp_alpha, interp_beta
from .ecf import ecf_estimate_all
from .mle import mle_estimate
import numpy as np
from scipy.stats import levy_stable

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