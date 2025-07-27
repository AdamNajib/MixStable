import numpy as np
from scipy.integrate import quad
from scipy.stats import levy_stable
from rpy2.robjects import FloatVector
from alpha_stable_mixture.r_interface import libstable4u, stabledist, stats
from rpy2.rinterface import RRuntimeError
from numba import njit, prange
from .r_interface import stats
from statsmodels.nonparametric.bandwidths import bw_silverman
from scipy.stats import wasserstein_distance


L_alpha, L_beta, L_delta, L_omega = [], [], [], []
M_w = []

def log_likelihood_mixture(params, data):
    """
    Compute the negative log-likelihood for a mixture of two stable distributions.
    """
    # Unpack 9 parameters: w + 2x (alpha, beta, scale, loc)
    w = params[0]
    a1, b1, s1, l1 = params[1:5]
    a2, b2, s2, l2 = params[5:9]

    if not (0 < w < 1 and 0.1 < a1 <= 2 and 0.1 < a2 <= 2 and -1 <= b1 <= 1 and -1 <= b2 <= 1 and s1 > 0 and s2 > 0):
        return np.inf  # invalid
    
    p1 = (a1, b1, s1, l1)
    p2 = (a2, b2, s2, l2)
    try:
        p1 = r_stable_pdf(data, *p1)
        p2 = r_stable_pdf(data, *p2)
        mix_pdf = w * p1 + (1 - w) * p2
        log_likelihood = np.sum(np.log(np.clip(mix_pdf, 1e-300, None)))
        return -log_likelihood  # for minimization
    except Exception as e:
        print("MLE error:", e)
        return np.inf

def neg_log_likelihood(params, X):
    alpha, beta, gamma, delta = params
    if not (0.1 < alpha <= 2 and -1 <= beta <= 1 and gamma > 0):
        return np.inf
    return -np.sum(levy_stable.logpdf(X, alpha, beta, loc=delta, scale=gamma))


def negative_log_likelihood(params, data):
    alpha, beta, gamma, delta = params
    if not (0 < alpha <= 2 and -1 <= beta <= 1 and gamma > 0):
        return np.inf
    try:
        pdf_vals = r_stable_pdf(data, alpha, beta, gamma, delta)
        log_likelihood = np.sum(np.log(np.clip(pdf_vals, 1e-300, None)))
        return -log_likelihood
    except Exception:
        return np.inf

def unpack_params(p):
    """
    Helper to unpack parameter dictionary into tuple.
    """
    return p['alpha'], p['beta'], p['gamma'], p['delta']

def ensure_positive_scale(scale, min_value=1e-6):
    return scale if scale > 0 else min_value

def estimate_bandwidth_custom(x):
    """
    Estimate kernel bandwidth using a second-order method.
    Based on Gaussian kernel density estimation.
    """

    if len(x) == 0:
        raise ValueError("Input array 'x' must not be empty.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Input array 'x' must contain only finite values.")

    n = len(x)
    C = min(np.std(x), (np.percentile(x, 75) - np.percentile(x, 25)) / 1.349)
    b_n = C * n ** (-2 / 5)
    b_n_prime = C * n ** (-3 / 5)

    I1 = 1 / (n * (n - 1) * b_n) * sum(
        np.sum(N_gaussien((x[i] - np.delete(x, i)) / b_n)) for i in range(n)
    )

    w2 = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                x_excluded = np.delete(x, [i, j])
                term1 = N_gaussien_2((x[i] - x[j]) / b_n_prime)
                term2 = np.sum(N_gaussien_2((x[i] - x_excluded) / b_n_prime))
                w2 += term1 * term2

    I2 = (1 / (n * (n - 1) * (n - 2) * b_n_prime ** 6)) * w2
    K1 = I1 / I2

    valeur1 = quad(lambda z: N_gaussien(z) ** 2, -np.inf, np.inf)[0]
    valeur2 = quad(lambda z: z ** 2 * N_gaussien(z), -np.inf, np.inf)[0] ** 2

    h_n = (valeur1 / valeur2) ** (1 / 5) * abs(K1) ** (1 / 5) * n ** (-1 / 5)
    return h_n

def estimate_bandwidth_r(x):
    """
    Computes bandwidth using R's bw.nrd0 and bw.SJ methods.
    """
    if len(x) == 0 or not np.all(np.isfinite(x)):
        raise ValueError("Input array must be valid and finite.")

    x_r = FloatVector(list(x))
    bw_nrd0 = stats.bw_nrd0(x_r)[0]
    bw_sj = stats.bw_SJ(x_r)[0]
    return bw_nrd0, bw_sj

def rstable_py(n, alpha, beta, scale=1.0, loc=0.0, pm=1):

    """
    Generate random samples from a stable distribution.

    Parameters:
        n (int): Number of samples to generate.
        alpha (float): Stability parameter (0 < alpha <= 2).
        beta (float): Skewness parameter (-1 <= beta <= 1).
        scale (float): Scale parameter (> 0). Default is 1.0.
        loc (float): Location parameter. Default is 0.0.
        pm (int): Parameterization mode (1 or 0). Default is 1.
                  - pm=1: S0 parameterization (default in scipy.stats.levy_stable).
                  - pm=0: S1 parameterization.

    Returns:
        np.ndarray: Array of random samples from the stable distribution.

    Raises:
        ValueError: If input parameters are invalid.
    """
    
    if not (0 < alpha <= 2):
        raise ValueError("Parameter 'alpha' must be in the range (0, 2].")
    if not (-1 <= beta <= 1):
        raise ValueError("Parameter 'beta' must be in the range [-1, 1].")
    if scale <= 0:
        raise ValueError("Parameter 'scale' must be greater than 0.")
    if pm not in [0, 1]:
        raise ValueError("Parameter 'pm' must be either 0 or 1.")

    # Adjust for S1 parameterization if pm=0
    if pm == 0:
        if alpha != 1:
            loc -= beta * scale * np.tan(np.pi * alpha / 2)
        else:
            loc -= beta * scale * (2 / np.pi) * np.log(scale)

    # Generate samples using scipy's levy_stable
    samples = levy_stable.rvs(alpha, beta, loc=loc, scale=scale, size=n)
    return samples

def rstable(n, alpha, beta, scale=1.0, loc=0.0, pm=1):
    """
    Generate random samples from a stable distribution using R's stabledist.rstable.

    Parameters:
        n (int): Number of samples to generate.
        alpha (float): Stability parameter (0 < alpha <= 2).
        beta (float): Skewness parameter (-1 <= beta <= 1).
        scale (float): Scale parameter (> 0). Default is 1.0.
        loc (float): Location parameter. Default is 0.0.
        pm (int): Parameterization mode (0 or 1). Default is 1.

    Returns:
        np.ndarray: Array of random samples from the stable distribution.
    """
    if not (0 < alpha <= 2):
        raise ValueError("Parameter 'alpha' must be in the range (0, 2].")
    if not (-1 <= beta <= 1):
        raise ValueError("Parameter 'beta' must be in the range [-1, 1].")
    if scale <= 0:
        raise ValueError("Parameter 'scale' must be greater than 0.")
    if pm not in [0, 1]:
        raise ValueError("Parameter 'pm' must be either 0 or 1.")

    try:
        # Call the R function
        samples = stabledist.rstable(n, alpha, beta, scale, loc, pm=pm)
        # Convert R vector to NumPy array
        samples = np.array(samples)
        return samples
    except Exception as e:
        raise RuntimeError(f"Unexpected error in rstable: {e}")

# 🔧 Stable PDF
def r_stable_pdf(x, alpha, beta, scale, location):
    """
    Computes the PDF of a stable distribution using the R library `libstable4u`.

    Parameters:
        x (array-like): Input data points.
        alpha (float): Stability parameter (0 < alpha <= 2).
        beta (float): Skewness parameter (-1 <= beta <= 1).
        scale (float): Scale parameter (> 0).
        location (float): Location parameter.

    Returns:
        np.ndarray: PDF values for the input data points.
    """
    try:
        # Validate inputs
        if not isinstance(x, (list, np.ndarray)):
            raise ValueError("Input 'x' must be a list or numpy array.")
        if not np.all(np.isfinite(x)):
            raise ValueError("Input 'x' contains non-finite values.")
        if not (0 < alpha <= 2):
            raise ValueError("Parameter 'alpha' must be in the range (0, 2].")
        if not (-1 <= beta <= 1):
            raise ValueError("Parameter 'beta' must be in the range [-1, 1].")
        if scale <= 0:
            raise ValueError("Parameter 'scale' must be greater than 0.")
        if not np.isfinite(location):
            raise ValueError("Parameter 'location' must be a finite number.")

        # Convert inputs to R-compatible types
        x = np.asarray(x, dtype=float)  
        x_r = FloatVector(x.tolist())
        pars = FloatVector([alpha,beta,scale,location])

        # Call the R function
        result = libstable4u.stable_pdf(x_r, pars)

        # Convert result back to NumPy array
        return np.array(result)

    except RRuntimeError as e:
        print(f"[R PDF Error] RRuntimeError: {e}")
        return np.zeros(len(x) if isinstance(x, (list, np.ndarray)) else 1)
    except ValueError as e:
        print(f"[Input Validation Error] {e}")
        return np.zeros(len(x) if isinstance(x, (list, np.ndarray)) else 1)
    except Exception as e:
        print(f"[Unexpected Error] {e}")
        return np.zeros(len(x) if isinstance(x, (list, np.ndarray)) else 1)
 
def stable_fit_init(x):
    """
    Initializes parameters for fitting a stable distribution.

    Parameters:
        x (array-like): Input data points.

    Returns:
        tuple: Estimated parameters (alpha, beta, gamma, delta).
    """
    if len(x) < 2:
        raise ValueError("Input data 'x' must contain at least 2 data points.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Input data 'x' must contain only finite values.")

    try:
        x_r = FloatVector(np.array(x).tolist())
        result = libstable4u.stable_fit_init(x_r)
        if len(result) < 4:
            raise ValueError("libstable4u.stable_fit_init returned fewer than 4 values.")
        alpha, beta, gamma, delta = result[:4]
    except Exception as e:
        print(f"Warning: Falling back to default parameters due to error: {e}")
        alpha, beta, gamma, delta = 1.5, 0.0, np.std(x), np.median(x)

    return alpha, beta, gamma, delta

# Gaussian Kernel Function
def N_gaussien(z):
    """ Standard Gaussian kernel function. """
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z ** 2)

def N_gaussien_2(z):
    """Second derivative of Gaussian density function."""
    return (1 / np.sqrt(2 * np.pi)) * (z ** 2 * np.exp(-z ** 2 / 2) - np.exp(-z ** 2 / 2))

# Uniform Kernel Functiondef N_uniform(z):
def N_uniform(z):
    """ Uniform kernel function. """
    return (1 / 2) * (np.abs(z) <= 1)

# Epanechnikov Kernel Function
def N_epanechnikov(z):
    """ Epanechnikov kernel function. """
    return 0.75 * (1 - z**2) * (np.abs(z) <= 1)

# Real Part of Integral
def Re(r, u, x, bn):
    """ Cosine component of the empirical characteristic function. """
    return np.cos(u * r) * N_gaussien((r - x) / bn)

# Imaginary Part of Integral
def Im(r, u, x, bn):
    """ Sine component of the empirical characteristic function. """
    return np.sin(u * r) * N_gaussien((r - x) / bn)

# Integrate the Real Component
def Int_Re(u, x, bn):
    """ Computes the integral of Re over the entire real line. """
    return quad(lambda r: Re(r, u, x, bn), -np.inf, np.inf)[0]

# Integrate the Imaginary Component

def Int_Im(u, x, bn):
    """ Computes the integral of Im over the entire real line. """
    return quad(lambda r: Im(r, u, x, bn), -np.inf, np.inf)[0]

def recursive_weight(l):
	"""
	Recursive weight function for ECF smoothing.
	"""
	return (2 / 3 + 0.05) / l

@njit(parallel=True)
def fast_integrate_numba(f_vals, r_vals):
    total = 0.0
    for i in prange(len(r_vals) - 1):
        total += 0.5 * (f_vals[i] + f_vals[i + 1]) * (r_vals[i + 1] - r_vals[i])
    return total

def fast_integrate(func, a=-6, b=6, N=100):
    r_vals = np.linspace(a, b, N)
    f_vals = func(r_vals)
    return fast_integrate_numba(f_vals, r_vals)

def ecf_fn(x, u, method='kernel'):
    """
    Compute the empirical characteristic function (ECF) using kernel or recursive smoothing.
    
    Parameters:
    - x: Input data array.
    - u: Array of frequency values.
    - method: Method to use for ECF calculation ('kernel' or 'recursive').

    Returns:
    - yr: Real part of the ECF.
    - yi: Imaginary part of the ECF.
    """
    if len(x) == 0 or len(u) == 0:
        raise ValueError("Input arrays 'x' and 'u' must be non-empty.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(u)):
        raise ValueError("Input arrays 'x' and 'u' must contain finite values.")

    # Bandwidth from R
    bn = stats.bw_SJ(FloatVector(x.tolist()))[0]
    g = np.zeros(len(u))
    h = np.zeros(len(u))

    if method == 'kernel':
        for i in range(len(x)):
            xi = x[i]
            for j in range(len(u)):
                uj = u[j]
                g[j] += fast_integrate(lambda r: np.cos(uj * r) * N_gaussien((r - xi) / bn))
                h[j] += fast_integrate(lambda r: np.sin(uj * r) * N_gaussien((r - xi) / bn))
        g /= len(x) * bn
        h /= len(x) * bn

    elif method == 'recursive':
        step_n = np.prod(1 - recursive_weight(np.arange(1, len(x) + 1)))
        for i in range(len(x)):
            xi = x[i]
            for j in range(len(u)):
                uj = u[j]
                weight = (np.prod(1 - recursive_weight(np.arange(1, i + 2))) ** -1) * recursive_weight(i + 1)
                g[j] += weight * fast_integrate(lambda r: np.cos(uj * r) * N_gaussien((r - xi) / bn))
                h[j] += weight * fast_integrate(lambda r: np.sin(uj * r) * N_gaussien((r - xi) / bn))
        g *= step_n / bn
        h *= step_n / bn

    else:
        raise ValueError(f"Invalid method: {method}. Choose 'kernel' or 'recursive'.")

    # Compute magnitude and phase
    yr = np.sqrt(g**2 + h**2)
    yi = np.arctan2(h, g)
    return yr, yi

def eta0(u, alpha, gamma, eps=0.05):
    """
    Computes the eta0 function used in ECF-based parameter estimation.

    Parameters:
    - u: frequency values
    - alpha: stability parameter
    - gamma: scale parameter
    - eps: tolerance for alpha ≈ 1

    Returns:
    - eta0 values
    """
    u = np.asarray(u)
    if abs(alpha - 1) < eps:
        y = (2 / np.pi) * u * np.log(gamma * np.abs(u))
        y[u == 0] = 0.0
    else:
        y = np.tan(np.pi * alpha / 2) * np.sign(u) * (
            (np.abs(gamma * u) - np.abs(gamma * u) ** alpha) / (gamma ** alpha)
        )
    return y

def eta_func(t, alpha, gamma):
    """Compute η(γ t | α; 0) as defined in the PDF (eq 3)."""
    t = np.asarray(t)
    if np.isclose(alpha, 1.0):
        return (2 / np.pi) * t * np.log(np.abs(gamma * t))
    else:
        return np.tan(np.pi * alpha / 2) * np.sign(t) * (np.abs(gamma)**(1 - alpha)) * (np.abs(t) - np.abs(t)**alpha)

def sanitize_and_convert_params(params, conversion_type="python_to_r"):
    """
    Sanitizes parameters and converts between Python and R formats.

    Parameters:
        params (list, tuple, or FloatVector): Input parameters [alpha, beta, scale, loc].
        conversion_type (str): Type of conversion ("python_to_r" or "r_to_python").

    Returns:
        list or FloatVector: Sanitized and converted parameters.
    """
    if conversion_type not in ["python_to_r", "r_to_python"]:
        raise ValueError("Invalid conversion_type. Choose 'python_to_r' or 'r_to_python'.")

    # Sanitize parameters
    if isinstance(params, (list, tuple, np.ndarray)):
        if len(params) != 4:
            raise ValueError("Input params must contain exactly 4 elements: [alpha, beta, scale, loc].")
        alpha, beta, scale, loc = params
        alpha = np.clip(alpha, 0.1, 2.0)
        beta = np.clip(beta, -1.0, 1.0)
        scale = max(scale, 1e-3)
        loc = float(loc) if np.isfinite(loc) else 0.0
        sanitized_params = [alpha, beta, scale, loc]
    elif conversion_type == "r_to_python" and isinstance(params, FloatVector):
        sanitized_params = list(params)
    else:
        raise ValueError("Input params must be a list, tuple, NumPy array, or FloatVector.")

    # Convert between Python and R formats
    if conversion_type == "python_to_r":
        from rpy2.robjects import FloatVector
        return FloatVector(sanitized_params)
    elif conversion_type == "r_to_python":
        return sanitized_params
    
def mixture_stable_pdf(x, p1, p2, w):
    y1 = r_stable_pdf(x, *p1)
    y2 = r_stable_pdf(x, *p2)
    return w * y1 + (1 - w) * y2

def kde_bandwidth_plugin(X, alpha):
    """
    Select KDE bandwidth based on plug-in method:
    - Sheather & Jones for α > 1
    - Slaoui's approach (here simulated as Silverman's) for α < 1

    Parameters:
        X : array-like
        alpha : float

    Returns:
        float : selected bandwidth
    """
    if alpha > 1:
        # Sheather & Jones (approximated via statsmodels Silverman)
        return bw_silverman(X)
    else:
        # Slaoui plug-in (approximated similarly)
        return bw_silverman(X) * (1.06 if alpha < 0.8 else 1.0)
    
def export_mse_to_latex(df, filename="mse_results.tex"):
    """
    Export the MSE DataFrame to LaTeX table.
    """
    with open(filename, "w") as f:
        f.write(df.to_latex(float_format="%.4f"))

def validate_params(alpha, beta, delta, r=None):
    """
    Validates input parameters for stability and constraints.
    """

    if delta <= 0 or alpha <= 0:
        raise ValueError("delta > 0 and alpha > 0 are required.")
    if r is not None and r <= 0:
        raise ValueError("r > 0 is required.")
    beta = np.clip(beta, -1, 1)
    return beta

def sine_weighted_exp_ralpha(r, x, alpha, beta, delta, omega):
    """
    Computes sine-weighted exponential term for gradient wrt alpha.
    """
    beta = validate_params(alpha, beta, delta, r)
    r_alpha = r ** alpha
    phase = ((x - omega) / delta) * r - beta * np.tan(np.pi * alpha / 2) * r_alpha
    return np.sin(phase) * r_alpha * np.exp(-np.clip(r_alpha, 0, 700))

def sine_log_weighted_exp_ralpha(r, x, alpha, beta, delta, omega):
    """
    Computes sine-log-weighted exponential term for gradient wrt alpha.
    """
    beta = validate_params(alpha, beta, delta, r)
    r_alpha = r ** alpha
    phase = ((x - omega) / delta) * r - beta * np.tan(np.pi * alpha / 2) * r_alpha
    return np.sin(phase) * np.log(r) * r_alpha * np.exp(-np.clip(r_alpha, 0, 700))

def cosine_log_weighted_exp_ralpha(r, x, alpha, beta, delta, omega):
    """
     Computes cosine-weighted exponential term for gradient wrt delta.
    """
    beta = validate_params(alpha, beta, delta, r)
    r_alpha = r ** alpha
    phase = ((x - omega) / delta) * r - beta * np.tan(np.pi * alpha / 2) * r_alpha
    return np.cos(phase) * np.log(r) * r_alpha * np.exp(-np.clip(r_alpha, 0, 700))

def cosine_exp_ralpha(r, x, alpha, beta, delta, omega):
    """
    Computes cosine exponential term for gradient wrt alpha.
    """
    beta = validate_params(alpha, beta, delta)
    r_alpha = r ** alpha
    phase = ((x - omega) / delta) * r - beta * np.tan(np.pi * alpha / 2) * r_alpha
    return np.cos(phase) * np.exp(-np.clip(r_alpha, 0, 700))

def sine_exp_ralpha(r, x, alpha, beta, delta, omega):
    """
    Computes sine exponential term for gradient wrt alpha.
    """
    beta = validate_params(alpha, beta, delta)
    r_alpha = r ** alpha
    phase = ((x - omega) / delta) * r - beta * np.tan(np.pi * alpha / 2) * r_alpha
    return np.sin(phase) * np.exp(-np.clip(r_alpha, 0, 700))

def sine_r_weighted_exp_ralpha(r, x, alpha, beta, delta, omega):
    """
    Computes sine-r-weighted exponential term for gradient wrt alpha.
    """
    beta = validate_params(alpha, beta, delta)
    r_alpha = r ** alpha
    phase = ((x - omega) / delta) * r - beta * np.tan(np.pi * alpha / 2) * r_alpha
    return np.sin(phase) * r * np.exp(-np.clip(r_alpha, 0, 700))

def integrate_function(f, x, alpha, beta, delta, omega):
    """
    Integrates the given function f over the range [0, 1e3] for each element in x.
    """
    if np.ndim(x) > 0:
        return np.array([quad(f, 0, 1e3, args=(xi, alpha, beta, delta, omega), limit=100)[0] for xi in x])
    else:
        return quad(f, 0, 1e3, args=(x, alpha, beta, delta, omega), limit=100)[0]

def integrate_sine_weighted(x, alpha, beta, delta, omega):
    """
    Integrates the sine-weighted exponential function.
    """
    return integrate_function(sine_weighted_exp_ralpha, x, alpha, beta, delta, omega)

def integrate_sine_log_weighted(x, alpha, beta, delta, omega):
    """
    Integrates the sine-log-weighted exponential function.
    """
    return integrate_function(sine_log_weighted_exp_ralpha, x, alpha, beta, delta, omega)

def integrate_cosine_log_weighted(x, alpha, beta, delta, omega):
    """
    Integrates the cosine-log-weighted exponential function.
    """
    return integrate_function(cosine_log_weighted_exp_ralpha, x, alpha, beta, delta, omega)

def integrate_cosine(x, alpha, beta, delta, omega):
    """
    Integrates the cosine exponential function.
    """
    return integrate_function(cosine_exp_ralpha, x, alpha, beta, delta, omega)

def integrate_sine(x, alpha, beta, delta, omega):
    """
    Integrates the sine exponential function.
    """
    return integrate_function(sine_exp_ralpha, x, alpha, beta, delta, omega)

def integrate_sine_r_weighted(x, alpha, beta, delta, omega):
    """
    Integrates the sine-r-weighted exponential function.
    """
    return integrate_function(sine_r_weighted_exp_ralpha, x, alpha, beta, delta, omega)

def grad_loglik_alpha(alpha, beta, delta, omega, x):
    """
    Computes the gradient of the log-likelihood with respect to alpha.
    """
    term1 = (np.pi * beta / (2 * np.cos(np.pi * alpha / 2) ** 2)) * integrate_sine_weighted(x, alpha, beta, delta, omega)
    term2 = beta * np.tan(np.pi * alpha / 2) * integrate_sine_log_weighted(x, alpha, beta, delta, omega)
    term3 = integrate_cosine_log_weighted(x, alpha, beta, delta, omega)
    return (1 / (np.pi * delta)) * (term1 + term2 - term3)

def normalized_grad_alpha(alpha, beta, delta, omega, x):
    """
    Computes the normalized gradient of the log-likelihood with respect to alpha.
    """
    grad_vals = np.array([grad_loglik_alpha(alpha, beta, delta, omega, xi) for xi in x])
    param = [alpha, beta, delta, omega]
    pdf_vals = np.maximum(r_stable_pdf(x, *param), 1e-300)
    return np.sum(grad_vals) / np.sum(pdf_vals)

def grad_loglik_beta(alpha, beta, delta, omega, x):
    """
    Computes the gradient of the log-likelihood with respect to beta.
    """
    return (np.tan(np.pi * alpha / 2) / (np.pi * delta)) * integrate_sine_weighted(x, alpha, beta, delta, omega)

def grad_loglik_delta(alpha, beta, delta, omega, x):
    """
    Computes the gradient of the log-likelihood with respect to delta.
    """
    term1 = (-1 / (np.pi * delta ** 2)) * integrate_cosine(x, alpha, beta, delta, omega)
    term2 = ((x - omega) / (np.pi * delta ** 3)) * integrate_sine_r_weighted(x, alpha, beta, delta, omega)
    return term1 + term2

def grad_loglik_omega(alpha, beta, delta, omega, x):
    """
    Computes the gradient of the log-likelihood with respect to omega.
    """
    return (1 / (np.pi * delta ** 2)) * integrate_sine_r_weighted(x, alpha, beta, delta, omega)

def normalized_objective_beta(X,beta):
    """
    Computes the normalized objective function for beta.
    """
    alpha = L_alpha[-1]
    delta = L_delta[-1]
    omega = L_omega[-1]

    params = [alpha, beta, delta, omega]

    pdf_vals = np.maximum(r_stable_pdf(X, *params), 1e-300)
    return np.sum(grad_loglik_beta(alpha, beta, delta, omega, X)) / np.sum(pdf_vals)

def normalized_objective_delta(X,delta):
    """
    Computes the normalized objective function for delta.
    """
    alpha = L_alpha[-1]
    beta = L_beta[-1]
    omega = L_omega[-1]

    params = [alpha, beta, delta, omega]

    pdf_vals = np.maximum(r_stable_pdf(X, *params), 1e-300)
    return np.sum(grad_loglik_delta(alpha, beta, delta, omega, X)) / np.sum(pdf_vals)

def normalized_objective_omega(X,omega):
    """
    Computes the normalized objective function for omega.
    """
    alpha = L_alpha[-1]
    beta = L_beta[-1]
    delta = L_delta[-1]

    params = [alpha, beta, delta, omega]

    pdf_vals = np.maximum(r_stable_pdf(X,*params), 1e-300)
    return np.sum(grad_loglik_omega(alpha, beta, delta, omega, X)) / np.sum(pdf_vals)

def false_position_update(a, b_n, f_a, f_b, objective_func):   
    """
    Performs a single iteration of the false-position method to find a root of the objective function.

    Parameters:
    - a, b: Interval bounds.
    - f_a, f_b: Function values at a and b.
    - objective_func: Callable function whose root is being sought.

    Returns:
    - Updated estimate for the root.
 """

    if abs(f_a - f_b) < 1e-10:
        return a

    c = (a * f_b - b_n * f_a) / (f_b - f_a)

    if not np.isfinite(c):
        return a
    try:
        f_c = objective_func(c)
    except Exception:
        return a

    return c if f_c * f_a < 0 else c

def wasserstein_distance_mixture(params1, params2, size=5000):
    def sample(params):
        weights = np.array([p['pi'] for p in params])
        weights /= weights.sum()
        samples = []
        for _ in range(size):
            i = np.random.choice(len(params), p=weights)
            p = params[i]
            s = levy_stable.rvs(p['alpha'], p['beta'], loc=p['delta'], scale=p['gamma'])
            samples.append(s)
        return np.array(samples)

    s1 = sample(params1)
    s2 = sample(params2)
    return wasserstein_distance(s1, s2)