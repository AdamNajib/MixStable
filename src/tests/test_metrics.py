import os
import numpy as np
from Mixstable import compute_model_metrics

def test_metrics():
    data = np.random.standard_cauchy(1000)
    data = data[(data > -25) & (data < 25)]

    params = (1.5, 0.0, 1.0, 0.0)
    result = compute_model_metrics(data, params)

    base_dir = os.path.dirname(__file__)
    result_dir = os.path.join(base_dir, "result", "metrics_result")
    os.makedirs(result_dir, exist_ok=True)

    output_path = os.path.join(result_dir, "metrics_result.txt")
    with open(output_path, "w") as f:
        for key, value in result.items():
            f.write(f"{key}: {value}\n")

    print("Metrics test completed. Output saved to:", output_path)

if __name__ == "__main__":
    test_metrics()
