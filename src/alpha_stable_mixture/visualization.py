# visualization.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .r_interface import libstable4u, stats
import pandas as pd
from rpy2.robjects import FloatVector
from .utils import r_stable_pdf, mixture_stable_pdf
from .gibbs import mock_gibbs_sampling
from scipy.stats import levy_stable
from scipy.interpolate import make_interp_spline

def plot_distributions(x, params_stable):
    sns.histplot(x, kde=True, stat="density", label="Data", bins=50, color='gray')
    xx = np.linspace(min(x), max(x), 500)

    # Normal PDF
    norm_pdf = stats.norm.pdf(xx, np.mean(x), np.std(x))
    plt.plot(xx, norm_pdf, 'r--', label="Normal")

    # Stable PDF from R
    pars = FloatVector([params_stable[k] for k in ["alpha", "beta", "gamma", "delta"]])
    xx_r = FloatVector(xx.tolist())
    try:
        pdf_vals = libstable4u.stable_pdf(xx_r, pars)
        plt.plot(xx, pdf_vals, 'b-', label="Stable (R)")
    except Exception as e:
        print(f"Stable PDF error: {e}")

    plt.legend()
    plt.title("Density Comparison")
    plt.show()

def plot_mixture(data, params1, params2, w, label="EM"):
    x = np.linspace(min(data), max(data), 1000)
    pdf1 = r_stable_pdf(x, *params1)
    pdf2 = r_stable_pdf(x, *params2)
    mix = w * pdf1 + (1 - w) * pdf2

    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=40, density=True, alpha=0.4, label='Data')
    plt.plot(x, mix, label=f'{label} Fit', lw=2)
    plt.plot(x, w * pdf1, '--', alpha=0.6, label=f'{label} Comp 1')
    plt.plot(x, (1 - w) * pdf2, '--', alpha=0.6, label=f'{label} Comp 2')
    plt.title(f"Mixture of Alpha-Stable Distributions ({label})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xlabel("Serial Interval (Days)")
    plt.ylabel("Density")
    plt.savefig(f"mixture_alpha_stable_{label}.png")  # Save the plot as an image file
    #plt.show()

def plot_final_mixture_fit(data, p1, p2, w):
    print(f"\nPlotting mixture:")
    print(f"  Weight: {w}")
    print(f"  Params1: {p1}")
    print(f"  Params2: {p2}")
    
    x_vals = np.linspace(min(data), max(data), 1000)
    
    try:
        y_vals = mixture_stable_pdf(x_vals, p1, p2, w)
    except Exception as e:
        print(f"[Error in PDF calculation]: {e}")
        return

    plt.figure(figsize=(9, 5))
    plt.hist(data, bins=40, density=True, alpha=0.6, color='gray', label='Data')
    plt.plot(x_vals, y_vals, color='red', linewidth=2, label='Fitted Mixture')
    plt.title("Fitted Mixture of Alpha-Stable Distributions")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("mixture_alpha_stable_fit_final.png")

def plot_method_comparison(scores):
    df = pd.DataFrame(scores).T.reset_index().rename(columns={'index': 'Method'})
    df = df.melt(id_vars='Method', var_name='Metric', value_name='Value')

    fig, ax1 = plt.subplots(1, 2, figsize=(12, 4))
    sns.barplot(data=df[df['Metric'] == 'RMSE'], x='Method', y='Value', ax=ax1[0], palette='viridis')
    ax1[0].set_title('RMSE per Method')
    ax1[0].tick_params(axis='x', rotation=45)

    sns.barplot(data=df[df['Metric'] == 'LogLikelihood'], x='Method', y='Value', ax=ax1[1], palette='magma')
    ax1[1].set_title('LogLikelihood per Method')
    ax1[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig("method_comparison.png")

def plot_trace(samples, param_name):
    for chain in samples:
        plt.plot(chain[param_name])
    plt.title(f"Trace plot for {param_name}")
    plt.show()

# Plotting function to visualize the results
def plot_results(M2_w1, M2_alpha1, M2_beta1, M2_delta1, M2_omega1, M2_w2, M2_alpha2, M2_beta2, M2_delta2, M2_omega2, xx, xx_true, yy_true):
    params1 = [np.mean(M2_alpha1[100:301]), np.mean(M2_beta1[100:301]), np.mean(M2_delta1[100:301]), np.mean(M2_omega1[100:301])] 
    params2 = [np.mean(M2_alpha2[100:301]), np.mean(M2_beta2[100:301]), np.mean(M2_delta2[100:301]), np.mean(M2_omega2[100:301])]
    yy = np.mean(M2_w1[100:301]) * r_stable_pdf(xx, *params1) + (1 - np.mean(M2_w1[100:301])) * r_stable_pdf(xx, *params2)

    plt.plot(xx, yy, label="Estimate",color='red', linestyle='--', lw=2)
    plt.plot(xx_true, yy_true, label="Truth", color='black', lw=2)
    plt.legend(loc="upper right")
    plt.title("Mixture of Alpha-Stable Distributions - Metropolis-Hastings")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("bayesian_stable_mixture_final.png")  # Save the plot as an image fileplot as an image file
    #plt.show()

# ----------------- Plot Comparison EM vs Non-Optimal -----------------
def plot_comparison(data, p1, p2, w):
    x = np.linspace(min(data), max(data), 1000)
    pdf1 = np.clip(r_stable_pdf(x, *p1), 1e-300, None)
    pdf2 = np.clip(r_stable_pdf(x, *p2), 1e-300, None)
    y_em = w * pdf1 + (1 - w) * pdf2

    # Non-optimized reference model

    params1 = [1.5, 0.0, 1.0, -1.0]
    params2 = [1.7, 0.5, 2.0, 6.0]

    y_bad = 0.5 * np.clip(r_stable_pdf(x, *params1), 1e-300, None) + \
                0.5 * np.clip(r_stable_pdf(x, *params2), 1e-300, None)

    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=40, density=True, alpha=0.5, label="Data", color="gray")
    plt.plot(x, y_em, label="EM Estimated", color="green", lw=2)
    plt.plot(x, y_bad, label="Non-Optimal", color="red", linestyle="--")
    plt.legend()
    plt.grid(True)
    plt.title("Alpha-Stable Mixture: EM Fit vs Non-Optimal")
    plt.tight_layout()
    plt.savefig("mixture_alpha_stable_comparison.png")  # Save the plot as an image file
    plt.xlabel("x")
    # plt.show()

def plot_mixture_fit(
    data,
    estimated_params,
    bins=200,
    plot_components=True,
    save_path=None,
    show_plot=True,
    title="Mixture of Alpha-Stable Distributions"
):
    """
    Plot the fitted alpha-stable mixture model against observed data.

    Parameters:
        data (array-like): Observed data points.
        estimated_params (dict): Dictionary with keys:
            - 'weights': list of weights (sum to 1)
            - 'alphas': list of alpha values
            - 'betas': list of beta values
            - 'gammas': list of scale (gamma) values
            - 'deltas': list of location (delta) values
        bins (int): Number of bins for the histogram.
        plot_components (bool): Whether to plot individual components.
        save_path (str or None): File path to save the plot (e.g., "fit.png").
        show_plot (bool): Whether to display the plot using plt.show().
        title (str): Title of the plot.
    """
    x = np.linspace(np.min(data), np.max(data), 1000)
    pdf_mix = np.zeros_like(x)

    num_components = len(estimated_params['weights'])
    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=bins, density=True, alpha=0.5, label="Observed Data", color='cornflowerblue')

    for i in range(num_components):
        w = estimated_params['weights'][i]
        a = estimated_params['alphas'][i]
        b = estimated_params['betas'][i]
        g = estimated_params['gammas'][i]
        d = estimated_params['deltas'][i]
        param = (a, b, g, d)

        pdf_i = r_stable_pdf(x, *param)
        pdf_mix += w * pdf_i

        if plot_components:
            plt.plot(x, w * pdf_i, '--', lw=1.5, label=f"Component {i+1}")

    # Final mixture plot
    plt.plot(x, pdf_mix, 'r-', lw=2, label="Estimated Mixture")

    plt.xlabel("x")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    if show_plot:
        plt.show()

    #plt.close()

def plot_mse_comparison(mse_df):
    """
    Visualize MSE comparisons between methods across configs.
    """
    df_melted = mse_df.reset_index().melt(id_vars='index', var_name='Method', value_name='MSE')
    df_melted.rename(columns={'index': 'Configuration'}, inplace=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, x='Configuration', y='MSE', hue='Method')
    plt.title("Mean Squared Error Comparison Across Estimation Methods")
    plt.ylabel("MSE")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_real_mixture_fit(X, result):
    """
    Plot histogram of real data with fitted alpha-stable components.
    """
    X_range = np.linspace(X.min(), X.max(), 500)
    pdf1 = levy_stable.pdf(X_range, result["params1"]["alpha"], result["params1"]["beta"],
                           loc=result["params1"]["delta"], scale=result["params1"]["gamma"])
    pdf2 = levy_stable.pdf(X_range, result["params2"]["alpha"], result["params2"]["beta"],
                           loc=result["params2"]["delta"], scale=result["params2"]["gamma"])

    mixture = result["lambda1"] * pdf1 + (1 - result["lambda1"]) * pdf2

    plt.figure(figsize=(10, 5))
    sns.histplot(X, bins=40, stat='density', color='lightgray', label='Data')
    plt.plot(X_range, pdf1, label='Component 1', linestyle='--')
    plt.plot(X_range, pdf2, label='Component 2', linestyle='--')
    plt.plot(X_range, mixture, label='Mixture Fit', color='black')
    plt.title("Real Dataset: Mixture of Two α-Stable Distributions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_effective_reproduction_number(GT, S, inc, dates, est_r0_ml, RT, output_file="RN_avec_dates_EM-ML.pdf"):
    """
    Estimates and plots the effective reproduction number Rt with smoothing.

    Parameters:
        GT (array-like): Generation times.
        S (array-like): Serial intervals.
        inc (array-like): Incidence data (daily case counts).
        dates (array-like): Corresponding dates (as pd.Series or datetime64).
        est_r0_ml (function): Function to estimate R0.
        RT (function): Function to compute Rt over time.
        output_file (str): Path to save the PDF plot.
    """
    # Align lengths
    min_len = min(len(GT), len(S))
    GT = GT[:min_len]
    S = S[:min_len]

    # Estimate R0
    est_r0_ml_empirical = est_r0_ml(GT, S)
    print(f"Estimated R0 (Empirical): {est_r0_ml_empirical:.4f}")

    # Compute Rt
    Rt = RT(inc, GT)
    rt_df = pd.DataFrame({'Date': dates, 'Rt': Rt})

    # Drop NaNs and Infs
    rt_df = rt_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Rt'])
    date_numeric = (rt_df['Date'] - rt_df['Date'].min()).dt.days

    if len(date_numeric) >= 4:
        # Interpolation
        spl = make_interp_spline(date_numeric, rt_df['Rt'], k=3)
        date_range_numeric = np.linspace(date_numeric.min(), date_numeric.max(), 500)
        Rt_smooth = spl(date_range_numeric)
        date_range = rt_df['Date'].min() + pd.to_timedelta(date_range_numeric, unit='D')

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(date_range, Rt_smooth, color='black', label='Smoothed Rt')
        plt.fill_between(date_range, Rt_smooth, color='red', alpha=0.2)
        plt.title('Effective Reproduction Number over Time')
        plt.xlabel('Time')
        plt.ylabel('Effective Reproduction Number')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        print(f"Plot saved as {output_file}")
    else:
        print("Not enough data points for cubic spline interpolation.")

def plot_fit_vs_true(true_params, est_params, data, bins=100):
    x = np.linspace(min(data), max(data), 1000)

    def mixture_pdf(x, params):
        y = np.zeros_like(x)
        for p in params:
            y += p['pi'] * levy_stable.pdf(x, p['alpha'], p['beta'], loc=p['delta'], scale=p['gamma'])
        return y

    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=bins, density=True, alpha=0.4, label='Data')

    plt.plot(x, mixture_pdf(x, true_params), 'g--', lw=2, label='True PDF')
    plt.plot(x, mixture_pdf(x, est_params), 'r-', lw=2, label='Estimated PDF')

    plt.title("True vs Estimated Mixture Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()