import numpy as np
import pandas as pd
from scipy.stats import levy_stable
import matplotlib.pyplot as plt

def generate_mixture_data(K=2, N=1000, seed=None):
    if seed is not None:
        np.random.seed(seed)

    weights = np.random.dirichlet(np.ones(K), 1)[0]

    params = []
    data = []
    for i in range(K):
        alpha = np.random.uniform(1.3, 1.9)
        beta = np.random.uniform(-1, 1)
        gamma = np.random.uniform(0.5, 2.0)
        delta = np.random.uniform(-2, 2)
        n_i = int(weights[i] * N)
        sample = levy_stable.rvs(alpha, beta, loc=delta, scale=gamma, size=n_i)
        data.append(sample)
        params.append({
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'delta': delta,
            'pi': weights[i]
        })

    return np.concatenate(data), params

# === Generate Samples from Mixture of Alpha-Stable Distributions ===
def generate_alpha_stable_mixture(weights, alphas, betas, gammas, deltas, size=1000, random_state=None):
    """
    Generate samples from a mixture of alpha-stable distributions using R's `r_stable_pdf`.
    """
    np.random.seed(random_state)
    n_components = len(weights)
    weights = np.array(weights)
    weights /= weights.sum() 
     
    samples = []
    labels = []
    for _ in range(size):
        component = np.random.choice(n_components, p=weights)
        sample = levy_stable.rvs(alphas[component], betas[component], loc=deltas[component], scale=gammas[component])
        samples.append(sample)
        labels.append(component)
    return np.array(samples), np.array(labels)

# ----------------- Synthetic Hard Dataset -----------------
def generate_synthetic_data(n=1000):
    np.random.seed(42)
    comp1 = levy_stable.rvs(alpha=1.2, beta=0.9, loc=-2, scale=0.8, size=n // 2)
    comp2 = levy_stable.rvs(alpha=1.8, beta=-0.5, loc=5, scale=1.5, size=n // 2)
    data = np.concatenate([comp1, comp2])
    np.random.shuffle(data)
    return data

def compute_serial_interval(filepath):
    data = pd.read_csv(filepath, sep=";", decimal=".", header=0)
    data['x.lb'] = pd.to_datetime(data['x.lb'], format="%d/%m/%Y")
    data['x.ub'] = pd.to_datetime(data['x.ub'], format="%d/%m/%Y")
    data['y'] = pd.to_datetime(data['y'], format="%d/%m/%Y")

    reference_date = pd.to_datetime("2020-01-01")
    data['x.lb_days'] = (data['x.lb'] - reference_date).dt.days
    data['x.ub_days'] = (data['x.ub'] - reference_date).dt.days
    data['y_days'] = (data['y'] - reference_date).dt.days

    data['SI'] = data['y_days'] - (data['x.ub_days'] + data['x.lb_days']) / 2
    plt.hist(data['SI'], bins=30, density=True, alpha=0.6, color='g')
    plt.title("Histogramme de l'intervalle sérial")
    plt.xlabel("Intervalle Sérial (jours)")
    plt.ylabel("Fréquence")
    plt.grid(True)
    #plt.show()

    return data['SI'].to_numpy()