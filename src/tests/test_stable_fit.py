import numpy as np
from scipy.stats import levy_stable
from alpha_stable_mixture.ecf_estimators import fit_stable_ecf
import matplotlib.pyplot as plt

def test_stable_estimation():
    np.random.seed(0)

    # True parameters
    alpha_true, beta_true = 1.7, 0.5
    gamma_true, delta_true = 1.0, 0.0

    # Generate synthetic data
    data = levy_stable.rvs(alpha_true, beta_true, loc=delta_true, scale=gamma_true, size=2000)
    u = np.linspace(-10, 10, 100)

    # Estimate parameters using ECF
    est = fit_stable_ecf(data, u)

    print("\n--- Stable Parameter Estimation ---")
    print(f"True:     alpha={alpha_true}, beta={beta_true}, gamma={gamma_true}, delta={delta_true}")
    print(f"Estimated: alpha={est['alpha']:.3f}, beta={est['beta']:.3f}, gamma={est['gamma']:.3f}, delta={est['delta']:.3f}")

    # Plot histogram and fitted PDF (optional)
    x = np.linspace(min(data), max(data), 500)
    plt.hist(data, bins=100, density=True, alpha=0.5, label='Data')
    pdf = levy_stable.pdf(x, est['alpha'], est['beta'], loc=est['delta'], scale=est['gamma'])
    plt.plot(x, pdf, 'r-', label='Fitted')
    plt.title("ECF-based α-Stable Fit")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test_stable_estimation()
