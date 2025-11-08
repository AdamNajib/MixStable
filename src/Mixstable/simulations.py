import numpy as np
from scipy.stats import levy_stable
import matplotlib.pyplot as plt
from .utils import r_stable_pdf
from Mixstable.gibbs import mock_gibbs_sampling
from sklearn.metrics import mean_squared_error, log_loss
from Mixstable.ecf_estimators import estimate_stable_kernel_ecf, estimate_stable_weighted_ols
from Mixstable.generate_sample import generate_mixture_data
from Mixstable.mle import mle_estimate
from Mixstable.mcculloch import mcculloch_lookup_estimate, build_mcculloch_interpolators, generate_mcculloch_table
import pandas as pd

def simulate_mixture(n, weights, params):
    samples = []
    for _ in range(n):
        k = np.random.choice(len(weights), p=weights)
        p = params[k]
        x = levy_stable.rvs(p['alpha'], p['beta'], loc=p['delta'], scale=p['gamma'])
        samples.append(x)
    return np.array(samples)

# Compare Methods with and without Gibbs Sampling
def compare_methods_with_gibbs(data, em_params, ecf_kernel_params, ecf_empirical_params):
    """
    Compare the results of methods with and without Gibbs sampling.
    """
    # Perform Gibbs sampling
    best_params, all_samples = mock_gibbs_sampling(data, n_samples=500)
    gibbs_mean = np.mean([params for _, params in all_samples], axis=0) if all_samples else np.zeros(9)

    # Extract parameters from Gibbs sampling
    gibbs_params = ([gibbs_mean[1], gibbs_mean[2], gibbs_mean[3], gibbs_mean[4]],
                    [gibbs_mean[5], gibbs_mean[6], gibbs_mean[7], gibbs_mean[8]],
                    gibbs_mean[0])

    # Define plotting x-axis
    x_vals = np.linspace(min(data), max(data), 1000)

    methods = {
        "EM Estimated": em_params,
        "ECF Kernel": ecf_kernel_params,
        "ECF Empirical": ecf_empirical_params,
        "Gibbs Sampling": gibbs_params
    }

    for name, (param1, param2, weight) in methods.items():
        if isinstance(param1, dict):
            # Safely get parameters from dict, with loc defaulted to 0
            loc1 = param1.get('loc', 0)
            loc2 = param2.get('loc', 0)
            scale1 = param1.get('scale', 1)
            scale2 = param2.get('scale', 1)

            param1 = (param1['alpha'], param1['beta'], scale1, loc1)
            param2 = (param2['alpha'], param2['beta'], scale2, loc2)
            
            pdf1 = r_stable_pdf(x_vals, *param1)
            pdf2 = r_stable_pdf(x_vals, *param2)
        else:
            # Assume tuple (alpha, beta, scale, loc)
            pdf1 = r_stable_pdf(x_vals, *param1)
            pdf2 = r_stable_pdf(x_vals, *param2)

        mixture_pdf = weight * pdf1 + (1 - weight) * pdf2

        # Plot
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=40, density=True, alpha=0.5, label="Data", color="gray")
        plt.plot(x_vals, mixture_pdf, label=f"{name} Fit", lw=2)
        plt.title(f"{name} vs Data")
        plt.xlabel("x")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"mixture_alpha_stable_{name.replace(' ', '_').lower()}.png")  # Save the plot
        #plt.show()  

def evaluate_fit(data, methods, x_vals):
    scores = {}
    for name, (p1, p2, w) in methods.items():
        pdf1 = r_stable_pdf(x_vals, *p1)
        pdf2 = r_stable_pdf(x_vals, *p2)
        pdf_mix = w * pdf1 + (1 - w) * pdf2
        
        hist, bin_edges = np.histogram(data, bins=50, range=(min(x_vals), max(x_vals)), density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        pdf_model = np.interp(bin_centers, x_vals, pdf_mix)

        rmse = np.sqrt(mean_squared_error(hist, pdf_model))
        hist_normalized = hist / np.sum(hist)  # Normalize histogram to represent probabilities
        pdf_model_normalized = pdf_model / np.sum(pdf_model)  # Normalize model PDF
        ll = -log_loss(hist_normalized + 1e-6, pdf_model_normalized + 1e-6, normalize=True)

        scores[name] = {'RMSE': rmse, 'LogLikelihood': ll}

    return scores

def compare_estimators_on_simulations(n_samples=1000, n_runs=30,interp_alpha=None, interp_beta= None):
    """
    Simulate data and compare MLE, ECF, and McCulloch estimators on alpha-stable distributions.

    Returns:
        pd.DataFrame: Table of performance metrics for each estimator and run.
    """
    results = []
    u = np.linspace(0.1, 1, 20)
    for seed in range(n_runs):
        # Simulate data and true parameters
        data, true_params = generate_mixture_data(1, n_samples, seed=seed)
        true = true_params[0] if isinstance(true_params, list) else true_params

        # MLE
        mle = mle_estimate(data)
        mse_mle = np.mean([(mle[k] - true[k])**2 for k in ['alpha', 'beta', 'gamma', 'delta']])

        # ECF
        ecf = estimate_stable_kernel_ecf(data, u)
        mse_ecf = np.mean([(ecf[k] - true[k])**2 for k in ['alpha', 'beta', 'gamma', 'delta']])

        # McCulloch
        mc = mcculloch_lookup_estimate(data,interp_alpha, interp_beta)
        mse_mc = np.mean([(mc[k] - true[k])**2 for k in ['alpha', 'beta', 'gamma', 'delta']])

        results.append({
            "seed": seed,
            "method": "MLE",
            "mse": mse_mle,
            **mle
        })
        results.append({
            "seed": seed,
            "method": "ECF",
            "mse": mse_ecf,
            **ecf
        })
        results.append({
            "seed": seed,
            "method": "McCulloch",
            "mse": mse_mc,
            **mc
        })

    df = pd.DataFrame(results)
    return df