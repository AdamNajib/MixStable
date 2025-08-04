import os
import numpy as np
from alpha_stable_mixture.gibbs import mock_gibbs_sampling, gibbs_sampler, metropolis_hastings
from alpha_stable_mixture.ecf_estimators import CDF

def test_mock_gibbs_sampling():
    data = np.random.standard_cauchy(1000)
    data = data[(data > -25) & (data < 25)]

    best_params, samples = mock_gibbs_sampling(data, n_samples=100, verbose=True)

    base_dir = os.path.dirname(__file__)
    result_dir = os.path.join(base_dir, "result", "gibbs_result")
    os.makedirs(result_dir, exist_ok=True)

    output_path = os.path.join(result_dir, "mock_gibbs_result.txt")
    with open(output_path, "w") as f:
        f.write("Best parameters:\n")
        f.write(str(best_params) + "\n\n")
        f.write("All samples:\n")
        for ll, params in samples:
            f.write(f"Log-likelihood: {ll:.2f}, Params: {params}\n")

    print("✅ mock_gibbs_sampling test completed. Output saved to:", output_path)

def test_gibbs_sampler():
    data = np.random.normal(loc=0, scale=1, size=500)

    samples = gibbs_sampler(data, iterations=100)

    base_dir = os.path.dirname(__file__)
    result_dir = os.path.join(base_dir, "result", "gibbs_result")
    os.makedirs(result_dir, exist_ok=True)

    output_path = os.path.join(result_dir, "gibbs_sampler_result.txt")
    with open(output_path, "w") as f:
        for s in samples:
            f.write(f"{s}\n")

    print("✅ gibbs_sampler test completed. Output saved to:", output_path)

def test_metropolis_hastings():
    data = np.random.standard_cauchy(1000)
    data = data[(data > -25) & (data < 25)]

    result = metropolis_hastings(fct=CDF, iterations=50, lok=data)

    base_dir = os.path.dirname(__file__)
    result_dir = os.path.join(base_dir, "result", "gibbs_result")
    os.makedirs(result_dir, exist_ok=True)

    output_path = os.path.join(result_dir, "metropolis_result.txt")
    with open(output_path, "w") as f:
        f.write("Metropolis-Hastings result:\n")
        for r in result:
            f.write(str(r) + "\n")

    print("✅ metropolis_hastings test completed. Output saved to:", output_path)

if __name__ == "__main__":
    test_mock_gibbs_sampling()
    test_gibbs_sampler()
    test_metropolis_hastings()
