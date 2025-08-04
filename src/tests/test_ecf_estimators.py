import os
import numpy as np
from alpha_stable_mixture.ecf_estimators import (
    CDF,
    ecf_regression,
    robust_ecf_regression,
    fit_stable_ecf,
    estimate_stable_recursive_ecf,
    estimate_stable_kernel_ecf,
    estimate_stable_weighted_ols,
    estimate_stable_from_cdf
)

def test_ecf_estimators():
    # Génération de données synthétiques
    data = np.random.standard_cauchy(1000)
    data = data[(data > -25) & (data < 25)]
    u = np.linspace(0.1, 1.0, 50)

    # Dossier de sortie
    base_dir = os.path.dirname(__file__)
    result_dir = os.path.join(base_dir, "result", "ecf_estimators_result")
    os.makedirs(result_dir, exist_ok=True)

    # Liste des fonctions à tester
    functions = {
        "CDF": lambda: CDF(data, u),
        "ecf_regression": lambda: ecf_regression(data, u),
        "robust_ecf_regression": lambda: robust_ecf_regression(data, u),
        "fit_stable_ecf": lambda: fit_stable_ecf(data, u),
        "estimate_stable_recursive_ecf": lambda: estimate_stable_recursive_ecf(data, u),
        "estimate_stable_kernel_ecf": lambda: estimate_stable_kernel_ecf(data, u),
        "estimate_stable_weighted_ols": lambda: estimate_stable_weighted_ols(data, u),
        "estimate_stable_from_cdf": lambda: estimate_stable_from_cdf(data, u)
    }

    for name, func in functions.items():
        try:
            result = func()
            output_path = os.path.join(result_dir, f"{name}_result.txt")
            with open(output_path, "w") as f:
                for key, value in result.items():
                    f.write(f"{key}: {value}\n")
            print(f"✅ {name} test completed. Output saved to:", output_path)
        except Exception as e:
            error_path = os.path.join(result_dir, f"{name}_error.txt")
            with open(error_path, "w") as f:
                f.write(f"❌ Error in {name}: {e}\n")
            print(f"❌ {name} test failed. Error saved to:", error_path)

if __name__ == "__main__":
    test_ecf_estimators()
