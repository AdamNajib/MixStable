import numpy as np
from alpha_stable_mixture.generate_sample import generate_mixture_data
from alpha_stable_mixture.utils import (
    wasserstein_distance_mixture,
    sample_from_mixture  #missinng
)
from alpha_stable_mixture.visualization import plot_mixture

def test_utils():
    data, params1 = generate_mixture_data(K=2, N=1000, seed=0)
    _, params2 = generate_mixture_data(K=2, N=1000, seed=1)

    print("\n--- Utility Function Testing ---")

    # Wasserstein distance
    d = wasserstein_distance_mixture(params1, params2)
    print(f"Wasserstein distance between mixtures: {d:.4f}")

    # Sampling from mixture
    samples = sample_from_mixture(params1, size=500)
    print(f"Sampled {len(samples)} points from mixture.")

    # Plot both mixtures
    plot_mixture(data, params1, title="Mixture 1")
    plot_mixture(data, params2, title="Mixture 2")

if __name__ == "__main__":
    test_utils()
