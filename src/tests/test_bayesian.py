import os
import numpy as np
from Mixstable import bayesian_mixture_model
import arviz as az
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="cupy._environment")


def test_bayesian():
    data = np.random.standard_t(df=1.5, size=500)
    model, trace = bayesian_mixture_model(data, draws=2000, chains=4)
    
    base_dir = os.path.dirname(__file__)
    result_dir = os.path.join(base_dir, "result", "bayesian_result")
    os.makedirs(result_dir, exist_ok=True)

    # Chemin complet du fichier de sortie
    output_path = os.path.join(result_dir, "bayesian_result")

    az.plot_trace(trace)
    summary = az.summary(trace)
    summary_path = os.path.join(result_dir, "bayesian_summary.txt")
    with open(summary_path, "w") as f:
        f.write(str(summary))
    print("📊 Summary saved to:", summary_path)

    with open(output_path, "w") as f:
        f.write(str(trace))
    print("✅ Bayesian test completed. Output saved to:", output_path)

if __name__ == "__main__":
    test_bayesian()
