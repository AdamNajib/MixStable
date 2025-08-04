import os
import numpy as np
from alpha_stable_mixture.utils import log_likelihood_mixture, rstable_py, stable_fit_init

def test_log_likelihood_mixture():
    data = np.random.standard_cauchy(1000)
    data = data[(data > -25) & (data < 25)]

    # Paramètres simulés : [w, α1, β1, γ1, δ1, α2, β2, γ2, δ2]
    params = [0.5, 1.5, 0.0, 1.0, 0.0, 1.8, 0.2, 1.2, 0.5]

    result = log_likelihood_mixture(params, data)

    base_dir = os.path.dirname(__file__)
    result_dir = os.path.join(base_dir, "result", "utils_result")
    os.makedirs(result_dir, exist_ok=True)

    output_path = os.path.join(result_dir, "log_likelihood_result.txt")
    with open(output_path, "w") as f:
        f.write(f"Negative Log-Likelihood: {result}\n")

    print("✅ log_likelihood_mixture test completed. Output saved to:", output_path)

if __name__ == "__main__":
    test_log_likelihood_mixture()

def test_rstable_py():
    samples = rstable_py(n=1000, alpha=1.5, beta=0.0, scale=1.0, loc=0.0, pm=1)

    base_dir = os.path.dirname(__file__)
    result_dir = os.path.join(base_dir, "result", "utils_result")
    os.makedirs(result_dir, exist_ok=True)

    output_path = os.path.join(result_dir, "rstable_py_samples.txt")
    with open(output_path, "w") as f:
        for val in samples:
            f.write(f"{val}\n")

    print("✅ rstable_py test completed. Output saved to:", output_path)

def test_stable_fit_init():
    data = np.random.standard_cauchy(1000)
    data = data[(data > -25) & (data < 25)]

    alpha, beta, gamma, delta = stable_fit_init(data)

    base_dir = os.path.dirname(__file__)
    result_dir = os.path.join(base_dir, "result", "utils_result")
    os.makedirs(result_dir, exist_ok=True)

    output_path = os.path.join(result_dir, "stable_fit_init_result.txt")
    with open(output_path, "w") as f:
        f.write(f"alpha: {alpha}\n")
        f.write(f"beta: {beta}\n")
        f.write(f"gamma: {gamma}\n")
        f.write(f"delta: {delta}\n")

    print("✅ stable_fit_init test completed. Output saved to:", output_path)


if __name__ == "__main__":
    test_rstable_py()
    test_log_likelihood_mixture()
    test_stable_fit_init()
