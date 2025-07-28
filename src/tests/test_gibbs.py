import numpy as np
from alpha_stable_mixture.generate_sample import generate_mixture_data
from alpha_stable_mixture.gibbs import gibbs_sampler_stable_mixture
from alpha_stable_mixture.utils import wasserstein_distance_mixture
import matplotlib.pyplot as plt
from alpha_stable_mixture.visualization import plot_mixture

def test_gibbs_fit():
    np.random.seed(123)
    K = 2
    N = 1000

    data, true_params = generate_mixture_data(K=K, N=N, seed=123)

    results = gibbs_sampler_stable_mixture(
        data=data,
        K=K,
        num_iterations=500,
        burn_in=100,
        thin=5,
        verbose=True
    )

    samples = results['samples']
    last_sample = samples[-1]

    print("\n--- Gibbs Sampling Summary ---")
    for i, (true, post) in enumerate(zip(true_params, last_sample)):
        print(f"\nComponent {i+1}:")
        print(f"  True: {true}")
        print(f"  Posterior (last sample): {post}")

    d = wasserstein_distance_mixture(true_params, last_sample)
    print(f"\nWasserstein distance (true vs posterior): {d:.4f}")

    plot_mixture(data, true_params, title="True Mixture")
    plot_mixture(data, last_sample, title="Gibbs Posterior Estimate")

if __name__ == "__main__":
    test_gibbs_fit()
