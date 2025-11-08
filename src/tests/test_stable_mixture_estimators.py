import os
import numpy as np
from Mixstable.stable_mixture_estimators import run_all_estimations, run_enzyme_estimations_with_gibbs

def test_run_all_estimations():
    data = np.random.standard_cauchy(1000)
    data = data[(data > -25) & (data < 25)]
    bw_sj = 0.5

    base_dir = os.path.dirname(__file__)
    root_result_dir = os.path.join(base_dir, "result", "stable_mixture_estimators_result")
    result_dir = os.path.join(root_result_dir, "run_all_estimations")
    os.makedirs(result_dir, exist_ok=True)

    try:
        # Rediriger les plots dans ce dossier
        os.chdir(result_dir)
        run_all_estimations(data, bw_sj, max_iter=100, tol=1e-4)
    
        with open(os.path.join(result_dir, "status.txt"), "w", encoding="utf-8") as f:
            f.write("✅ run_all_estimations executed successfully.\n")
    except Exception as e:
        with open(os.path.join(result_dir, "error.txt"), "w", encoding="utf-8") as f:
            f.write(f"❌ Error during run_all_estimations: {e}\n")


    print("✅ run_all_estimations test completed.")

def test_run_enzyme_estimations_with_gibbs():
    enzyme_data = np.random.standard_cauchy(1000)
    enzyme_data = enzyme_data[(enzyme_data > -25) & (enzyme_data < 25)]
    bw_sj = 0.5

    base_dir = os.path.dirname(__file__)
    root_result_dir = os.path.join(base_dir, "result", "stable_mixture_estimators_result")
    result_dir = os.path.join(root_result_dir, "run_enzyme_estimations_with_gibbs")
    os.makedirs(result_dir, exist_ok=True)

    try:
        os.chdir(result_dir)
        run_enzyme_estimations_with_gibbs(enzyme_data, bw_sj, max_iter=100, tol=1e-4)
        with open(os.path.join(result_dir, "status.txt"), "w", encoding="utf-8") as f:
            f.write("✅ run_all_estimations executed successfully.\n")
    except Exception as e:
        with open(os.path.join(result_dir, "error.txt"), "w", encoding="utf-8") as f:
            f.write(f"❌ Error during run_all_estimations: {e}\n")

    print("✅ run_enzyme_estimations_with_gibbs test completed.")

if __name__ == "__main__":
    test_run_all_estimations()
    test_run_enzyme_estimations_with_gibbs()
