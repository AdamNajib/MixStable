import numpy as np
from alpha_stable_mixture.em import em_stable_mixture
from alpha_stable_mixture.ecf_estimators import estimate_stable_weighted_ols
from alpha_stable_mixture.em import em_fit_alpha_stable_mixture
from alpha_stable_mixture.generate_sample import generate_mixture_data
from alpha_stable_mixture.utils import wasserstein_distance_mixture
from alpha_stable_mixture.visualization import plot_fit_vs_true

def test_em_stable_mixture_runs():
    np.random.seed(42)
    data = np.random.standard_normal(50)
    u = np.linspace(0.1, 1.5, 20)

    result = em_stable_mixture(data, u, estimator_func=estimate_stable_weighted_ols, max_iter=10)

    assert isinstance(result, dict), "Result should be a dictionary"
    assert "weights" in result and "params1" in result and "params2" in result, "Missing keys in result"
    assert 0 <= result["weights"] <= 1, "Invalid mixture weight"
    assert len(result["params1"]) == 4 and len(result["params2"]) == 4, "Invalid parameter lengths"
    print("✅ EM algorithm test passed.")

def test_em_fit():
    np.random.seed(42)
    K = 2
    N = 2000

    data, true_params = generate_mixture_data(K=K, N=N, seed=42)
    est_params, log_likelihoods = em_fit_alpha_stable_mixture(data, K=K, max_iter=100)

    print("\n--- EM Fitting Summary ---")
    for i, (true, est) in enumerate(zip(true_params, est_params)):
        print(f"\nComponent {i+1}:")
        print(f"  True: {true}")
        print(f"  Estimated: {est}")

    d = wasserstein_distance_mixture(true_params, est_params)
    print(f"\nWasserstein distance (true vs est): {d:.4f}")

    plot_fit_vs_true(true_params, est_params, data)

if __name__ == "__main__":
    test_em_stable_mixture_runs()
