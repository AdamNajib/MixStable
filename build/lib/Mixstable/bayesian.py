# bayesian.py
import numpy as np
import pymc as pm

def bayesian_mixture_model(data, draws=1000, chains=2):
    """
    Bayesian estimation of a two-component mixture of alpha-stable-like distributions
    using Student-T distributions as proxies.

    Parameters:
    - data: array-like, observed data
    - draws: int, number of posterior samples
    - chains: int, number of MCMC chains

    Returns:
    - model: PyMC model object
    - trace: InferenceData object containing posterior samples
    """
    with pm.Model() as model:
        # Mixture weights
        w = pm.Beta('w', 1., 1.)

        # Component 1
        alpha1 = pm.Uniform('alpha1', lower=0.1, upper=2)
        beta1 = pm.Uniform('beta1', lower=-1, upper=1)
        sigma1 = pm.HalfNormal('sigma1', sigma=2)
        mu1 = pm.Normal('mu1', mu=np.mean(data), sigma=5)

        # Component 2
        alpha2 = pm.Uniform('alpha2', lower=0.1, upper=2)
        beta2 = pm.Uniform('beta2', lower=-1, upper=1)
        sigma2 = pm.HalfNormal('sigma2', sigma=2)
        mu2 = pm.Normal('mu2', mu=np.mean(data), sigma=5)

        # Observations modeled as a mixture of Student-T distributions
        y_obs = pm.Mixture(
            'y_obs',
            w=[w, 1 - w],
            comp_dists=[
                pm.StudentT.dist(nu=alpha1, mu=mu1, sigma=sigma1),
                pm.StudentT.dist(nu=alpha2, mu=mu2, sigma=sigma2)
            ],
            observed=data
        )

        trace = pm.sample(draws=draws, chains=chains, return_inferencedata=True)

    return model, trace
