
import os
import numpy as np
from Mixstable import ecf_estimate_all

def test_ecf():
    data = np.random.standard_cauchy(1000)
    data = data[(data > -25) & (data < 25)]

    result = ecf_estimate_all(data)

    base_dir = os.path.dirname(__file__)
    result_dir = os.path.join(base_dir, "result", "ecf_result")
    os.makedirs(result_dir, exist_ok=True)

    output_path = os.path.join(result_dir, "ecf_result.txt")
    with open(output_path, "w") as f:
        for key, value in result.items():
            f.write(f"{key}: {value}\n")

    print("ECF test completed. Output saved to:", output_path)

if __name__ == "__main__":
    test_ecf()
