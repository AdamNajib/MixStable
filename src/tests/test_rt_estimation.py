import os
import numpy as np
from Mixstable.rt_estimation import RT, est_r0_ml, empirical_r0

def test_rt():
    incidence = np.random.poisson(lam=5, size=30)
    generation_time = np.random.gamma(shape=2, scale=1, size=10)
    rt_values = RT(incidence, generation_time)

    base_dir = os.path.dirname(__file__)
    result_dir = os.path.join(base_dir, "result", "rt_estimation_result")
    os.makedirs(result_dir, exist_ok=True)

    output_path = os.path.join(result_dir, "rt_result.txt")
    with open(output_path, "w") as f:
        for i, val in enumerate(rt_values):
            f.write(f"Day {i}: Rt = {val:.4f}\n")

    print("RT test completed. Output saved to:", output_path)

def test_est_r0_ml():
    incidence = np.random.poisson(lam=5, size=30)
    generation_time = np.random.gamma(shape=2, scale=1, size=10)
    r0 = est_r0_ml(generation_time, incidence)

    base_dir = os.path.dirname(__file__)
    result_dir = os.path.join(base_dir, "result", "rt_estimation_result")
    os.makedirs(result_dir, exist_ok=True)

    output_path = os.path.join(result_dir, "r0_ml_result.txt")
    with open(output_path, "w") as f:
        f.write(f"Estimated R0 (ML): {r0:.4f}\n")

    print("est_r0_ml test completed. Output saved to:", output_path)


def test_empirical_r0():
    incidence = np.random.poisson(lam=5, size=30)
    serial_interval = 4.0
    r0 = empirical_r0(incidence, serial_interval, growth_model="exponential")

    base_dir = os.path.dirname(__file__)
    result_dir = os.path.join(base_dir, "result", "rt_estimation_result")
    os.makedirs(result_dir, exist_ok=True)

    output_path = os.path.join(result_dir, "empirical_r0_result.txt")
    with open(output_path, "w") as f:
        f.write(f"Estimated R0 (empirical): {r0:.4f}\n")

    print("empirical_r0 test completed. Output saved to:", output_path)

if __name__ == "__main__":
    test_rt()
    test_est_r0_ml()
    test_empirical_r0()
