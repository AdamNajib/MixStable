import os
import numpy as np
from Mixstable.mcculloch import mcculloch_quantile_init, generate_mcculloch_table, build_mcculloch_interpolators, mcculloch_lookup_estimate

def test_mcculloch_quantile_init():
    data = np.random.standard_cauchy(1000)
    data = data[(data > -25) & (data < 25)]

    result = mcculloch_quantile_init(data)

    base_dir = os.path.dirname(__file__)
    result_dir = os.path.join(base_dir, "result", "mcculloch_result")
    os.makedirs(result_dir, exist_ok=True)

    output_path = os.path.join(result_dir, "mcculloch_init_result.txt")
    with open(output_path, "w") as f:
        for key, value in result.items():
            f.write(f"{key}: {value}\n")

    print("McCulloch quantile init test completed. Output saved to:", output_path)

def test_mcculloch_lookup_estimate():
    alpha_grid = np.linspace(0.6, 2.0, 5)
    beta_grid = np.linspace(-1.0, 1.0, 5)

    table = generate_mcculloch_table(alpha_grid, beta_grid, size=5000)
    interp_alpha, interp_beta = build_mcculloch_interpolators(table)

    data = np.random.standard_cauchy(1000)
    data = data[(data > -25) & (data < 25)]

    result = mcculloch_lookup_estimate(data, interp_alpha, interp_beta)

    base_dir = os.path.dirname(__file__)
    result_dir = os.path.join(base_dir, "result", "mcculloch_result")
    os.makedirs(result_dir, exist_ok=True)

    output_path = os.path.join(result_dir, "mcculloch_lookup_result.txt")
    with open(output_path, "w") as f:
        for key, value in result.items():
            f.write(f"{key}: {value}\n")

    print("McCulloch lookup estimate test completed. Output saved to:", output_path)

if __name__ == "__main__":
    test_mcculloch_lookup_estimate()
    test_mcculloch_quantile_init()
