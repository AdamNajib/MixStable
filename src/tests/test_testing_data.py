import os
import numpy as np
from alpha_stable_mixture.testing_data import analyse_stable_distribution

def test_utils():
    x = np.random.standard_cauchy(1000)
    x = x[(x > -25) & (x < 25)]

    # Dossier "result" situé au même niveau que ce fichier
    base_dir = os.path.dirname(__file__)
    result_dir = os.path.join(base_dir, "result", "serial_interval_output")
    os.makedirs(result_dir, exist_ok=True)

    # Chemin complet du fichier de sortie
    output_path = os.path.join(result_dir, "serial_interval_result")
    fig_path = os.path.join(result_dir, "serial_interval_plot.png")

    
    result_text, saved_fig = analyse_stable_distribution(x, filename=output_path, fig_path=fig_path)
    print("Ran OK:", result_text)

    assert os.path.exists(saved_fig), f"Figure not found at {saved_fig}"


if __name__ == "__main__":
    test_utils()
