# interface/em_robust.py
"""
Robust interface wrapper for EM algorithms in Streamlit
This only provides interface wrappers, doesn't modify the original alpha_stable_mixture package
"""

import numpy as np
from interface.rpy2_init import run_with_r_context, check_r_availability, ensure_rpy2_context
import traceback
import threading

def get_fallback_params(data):
    """Get fallback stable distribution parameters using method of moments"""
    data = np.asarray(data)
    
    # Remove outliers (keep within 3 standard deviations)
    mean_data = np.mean(data)
    std_data = np.std(data)
    clean_data = data[np.abs(data - mean_data) <= 3 * std_data]
    
    if len(clean_data) < 10:
        clean_data = data
    
    # Basic parameter estimation
    location = np.median(clean_data)  # delta (location)
    scale = np.std(clean_data) * 0.8  # gamma (scale), slightly reduced
    
    # Estimate alpha from kurtosis (rough approximation)
    kurt = np.mean((clean_data - np.mean(clean_data))**4) / np.var(clean_data)**2
    if kurt > 10:
        alpha = 1.2  # Heavy tails
    elif kurt > 5:
        alpha = 1.5
    else:
        alpha = 1.8  # Lighter tails
    
    # Beta (skewness) - simple approximation
    skew = np.mean((clean_data - np.mean(clean_data))**3) / np.var(clean_data)**1.5
    beta = np.clip(skew * 0.3, -0.8, 0.8)  # Keep reasonable bounds
    
    return [alpha, beta, scale, location]

def safe_em_wrapper(em_func, data, max_iter=100, tol=1e-4):
    """
    Thread-safe wrapper for EM functions from alpha_stable_mixture.em_methode
    """
    def em_call():
        # Ensure R context is properly set up for this thread
        if not ensure_rpy2_context():
            raise RuntimeError("Could not establish R context")
        
        print(f"🔧 Thread {threading.current_thread().name}: Calling {em_func.__name__}")
        return em_func(data, max_iter=max_iter, tol=tol)
    
    try:
        # Run with proper R context
        return run_with_r_context(em_call)
    except Exception as e:
        print(f"❌ EM function {em_func.__name__} failed: {e}")
        if "Conversion rules" in str(e):
            print("🔄 RPy2 context issue detected, using fallback...")
        raise e

def create_stable_mixture_result(data, params1, params2, weight, method_name):
    """Create standardized result dictionary"""
    try:
        # Try to calculate log-likelihood using the original method
        def calc_ll():
            from alpha_stable_mixture.utils import r_stable_pdf, unpack_params
            pdf1 = np.maximum(r_stable_pdf(data, *unpack_params(params1)), 1e-300)
            pdf2 = np.maximum(r_stable_pdf(data, *unpack_params(params2)), 1e-300)
            return np.sum(np.log(weight * pdf1 + (1 - weight) * pdf2))
        
        log_likelihood = run_with_r_context(calc_ll)
        
    except Exception as e:
        print(f"⚠️ Could not calculate log-likelihood: {e}")
        # Simple fallback
        log_likelihood = -len(data) * np.log(np.std(data)) - len(data) / 2
    
    return {
        "params_list": [params1, params2],
        "weights": [float(weight), float(1 - weight)],  # Convert to float, not numpy types
        "log_likelihood": float(log_likelihood),
        "method": method_name
    }

def robust_em_stable_mixture(data, u=None, estimator_func=None, max_iter=100, epsilon=1e-3):
    """
    Robust wrapper for EM stable mixture estimation that handles Streamlit threading issues
    
    This function calls the original EM functions from alpha_stable_mixture.em_methode
    but with proper RPy2 context management for Streamlit.
    """
    
    print(f"🔄 Starting robust EM wrapper for Streamlit...")
    
    # Validate and clean data
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    
    if len(data) == 0:
        raise ValueError("No valid data points")
    
    print(f"📊 Processing {len(data)} data points")
    
    # Map estimator functions to their corresponding EM functions
    method_mapping = {
        'estimate_stable_recursive_ecf': 'recursive_ecf',
        'estimate_stable_kernel_ecf': 'kernel_ecf',
        'estimate_stable_weighted_ols': 'weighted_ols',
        'estimate_stable_from_cdf': 'from_cdf'
    }
    
    # Determine which method to use
    if estimator_func is not None:
        func_name = estimator_func.__name__
        if func_name.endswith('_wrapped'):
            func_name = func_name[:-8]
        method_key = method_mapping.get(func_name, 'kernel_ecf')
    else:
        method_key = 'kernel_ecf'  # Default
    
    print(f"🔧 Using method: {method_key}")
    
    # Check R availability
    r_available, r_status = check_r_availability()
    print(f"R Status: {r_status}")
    
    if not r_available:
        print("❌ R interface not available, using fallback")
        return simple_mixture_fallback(data)
    
    # Import the EM functions from the original package
    try:
        from alpha_stable_mixture.em_methode import (
            em_estimate_stable_recursive_ecf,
            em_estimate_stable_kernel_ecf,
            em_estimate_stable_weighted_ols,
            em_estimate_stable_from_cdf
        )
        
        em_functions = {
            'recursive_ecf': em_estimate_stable_recursive_ecf,
            'kernel_ecf': em_estimate_stable_kernel_ecf,
            'weighted_ols': em_estimate_stable_weighted_ols,
            'from_cdf': em_estimate_stable_from_cdf
        }
        
        em_func = em_functions[method_key]
        
    except ImportError as e:
        print(f"❌ Could not import EM functions: {e}")
        return simple_mixture_fallback(data)
    
    # Try the original EM algorithm with proper context
    try:
        print(f"🔄 Attempting original EM algorithm: {em_func.__name__}")
        
        # Call with our safe wrapper
        params1, params2, weight = safe_em_wrapper(em_func, data, max_iter, epsilon)
        
        print(f"✅ Original EM succeeded with weight: {weight:.3f}")
        
        # Create result
        result = create_stable_mixture_result(
            data, params1, params2, weight, f"original_{method_key}"
        )
        
        return result
        
    except Exception as e:
        print(f"❌ Original EM failed: {e}")
        print("🔄 Falling back to simple mixture estimation...")
        return simple_mixture_fallback(data)

def simple_mixture_fallback(data):
    """Simple fallback mixture estimation using K-means clustering"""
    print("🔄 Using simple mixture fallback...")
    
    try:
        from sklearn.cluster import KMeans
        
        # Use K-means for initial clustering
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data.reshape(-1, 1))
        
        # Calculate weights
        weight = np.mean(labels == 0)
        
        # Get cluster data
        cluster0_data = data[labels == 0]
        cluster1_data = data[labels == 1]
        
        # Get parameters for each cluster
        if len(cluster0_data) >= 5:
            params1 = get_fallback_params_dict(cluster0_data)
        else:
            params1 = get_fallback_params_dict(data)
            
        if len(cluster1_data) >= 5:
            params2 = get_fallback_params_dict(cluster1_data)
        else:
            params2 = get_fallback_params_dict(data)
        
        # Simple log-likelihood approximation
        log_likelihood = -len(data) * np.log(np.std(data)) - len(data) / 2
        
        return {
            "params_list": [params1, params2],
            "weights": [float(weight), float(1 - weight)],  # Convert to plain Python float
            "log_likelihood": float(log_likelihood),
            "method": "simple_kmeans_fallback"
        }
        
    except Exception as e:
        print(f"❌ K-means fallback failed: {e}")
        return quantile_split_fallback(data)

def get_fallback_params_dict(data):
    """Convert fallback parameters to dictionary format"""
    params_list = get_fallback_params(data)
    return {
        'alpha': params_list[0],
        'beta': params_list[1], 
        'gamma': params_list[2],
        'delta': params_list[3]
    }

def quantile_split_fallback(data):
    """Ultimate fallback using quantile splitting"""
    print("🔄 Using quantile split fallback...")
    
    # Split at median
    median = np.median(data)
    
    # First component: lower half
    lower_data = data[data <= median]
    params1 = get_fallback_params_dict(lower_data if len(lower_data) > 0 else data)
    
    # Second component: upper half  
    upper_data = data[data > median]
    params2 = get_fallback_params_dict(upper_data if len(upper_data) > 0 else data)
    
    # Equal weights
    weight = 0.5
    
    # Simple log-likelihood
    log_likelihood = -len(data) * np.log(np.std(data)) - len(data) / 2
    
    return {
        "params_list": [params1, params2],
        "weights": [float(weight), float(1 - weight)],  # Convert to plain Python float  
        "log_likelihood": float(log_likelihood),
        "method": "quantile_split_fallback"
    }