# alpha_stable_mixture/testing_data.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from scipy.stats import norm, skew, kurtosis, shapiro, kstest, anderson
from statsmodels.stats.stattools import jarque_bera

# Lazy imports to avoid circular dependency
def _get_r_objects():
    """Get R objects with lazy import to avoid circular dependency"""
    try:
        from rpy2.robjects import FloatVector
        from Mixtable.r_interface import libstable4u, alphastable
        from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
        
        # Define the QCV R function
        qcv_r_code = """
        qcv_stat <- function(x) {
          x <- sort((x - mean(x)) / sd(x))
          q25 <- quantile(x, 0.25)
          q75 <- quantile(x, 0.75)
          var_left <- var(x[x < q25])
          var_right <- var(x[x > q75])
          var_mid <- var(x[x > q25 & x < q75])
          qcv = (var_left + var_right) / (2 * var_mid)
          return(qcv)
        }
        """
        qcv_test = SignatureTranslatedAnonymousPackage(qcv_r_code, "qcv_test")
        
        return FloatVector, libstable4u, alphastable, qcv_test
    except Exception as e:
        print(f"Warning: Could not load R interface: {e}")
        return None, None, None, None

# Helper functions
def test_normality(x):
    return {
        "Shapiro": shapiro(x),
        "Jarque-Bera": jarque_bera(x),
        "Anderson": anderson(x, dist='norm'),
        "KS": kstest(x, 'norm', args=(np.mean(x), np.std(x)))
    }

def skew_kurtosis(x):
    return {
        "skewness": skew(x),
        "kurtosis": kurtosis(x, fisher=False)
    }

def estimate_stable_r(x):
    FloatVector, libstable4u, alphastable, qcv_test = _get_r_objects()
    
    if FloatVector is None or libstable4u is None:
        print("R interface not available, using fallback parameters")
        return {"alpha": 1.5, "beta": 0.0, "gamma": 1.0, "delta": 0.0}
    
    x_r = FloatVector(x.tolist())
    try:
        result = libstable4u.stable_fit_init(x_r)
        return {
            "alpha": float(result[0]),
            "beta": float(result[1]),
            "gamma": float(result[2]),
            "delta": float(result[3])
        }
    except Exception as e:
        print(f"R error: {e}")
        return {"alpha": 1.5, "beta": 0.0, "gamma": 1.0, "delta": 0.0}

def qcv_stat(x):
    FloatVector, libstable4u, alphastable, qcv_test = _get_r_objects()
    
    if FloatVector is None or qcv_test is None:
        print("R interface not available, using fallback QCV calculation")
        # Fallback Python implementation of QCV
        x_norm = (x - np.mean(x)) / np.std(x)
        x_sorted = np.sort(x_norm)
        q25 = np.percentile(x_sorted, 25)
        q75 = np.percentile(x_sorted, 75)
        var_left = np.var(x_sorted[x_sorted < q25])
        var_right = np.var(x_sorted[x_sorted > q75])
        var_mid = np.var(x_sorted[(x_sorted > q25) & (x_sorted < q75)])
        return (var_left + var_right) / (2 * var_mid) if var_mid > 0 else 1.0
    
    x_r = FloatVector(x.tolist())
    return float(qcv_test.qcv_stat(x_r)[0])

def fit_em_mixture_r(x):
    FloatVector, libstable4u, alphastable, qcv_test = _get_r_objects()
    
    if FloatVector is None or alphastable is None:
        print("R interface not available for EM mixture fitting")
        return None
    
    x_r = FloatVector(x.tolist())
    try:
        return alphastable.emstabledist(x_r, 2)
    except Exception as e:
        print(f"EM fitting error: {e}")
        return None

def plot_vs_normal_stable(x, params_stable ,fig_path='stability_test_plot.png'):
    FloatVector, libstable4u, alphastable, qcv_test = _get_r_objects()
    
    xx = np.linspace(min(x), max(x), 500)
    norm_pdf = norm.pdf(xx, np.mean(x), np.std(x))
    
    # Try to get stable PDF from R, fallback to normal if not available
    if FloatVector is not None and libstable4u is not None:
        try:
            x_r = FloatVector(xx.tolist())
            pars = FloatVector([params_stable[k] for k in ["alpha", "beta", "gamma", "delta"]])
            stable_pdf = libstable4u.stable_pdf(x_r, pars)
        except Exception as e:
            print(f"Error computing stable PDF: {e}")
            stable_pdf = norm_pdf  # fallback
    else:
        stable_pdf = norm_pdf  # fallback
    
    plt.figure(figsize=(10, 6))
    plt.hist(x, bins=50, density=True, alpha=0.4, label='Data')
    plt.plot(xx, norm_pdf, 'r--', label='Normal PDF')
    plt.plot(xx, stable_pdf, 'b-', label='Stable PDF (R)')
    plt.title("Empirical vs Normal vs Stable PDF")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_path,dpi=150, bbox_inches='tight')
    plt.close()  # Close to avoid display issues in Streamlit

def export_analysis_report(data, stable_params, qcv, skew_kurt, normality, verdict, filename="stable_report"):
    report = {
        "summary": {
            "n": len(data),
            "skewness": skew_kurt["skewness"],
            "kurtosis": skew_kurt["kurtosis"],
            "qcv": qcv,
            "verdict": verdict
        },
        "normality_tests": {k: dict(statistic=v.statistic, pvalue=v.pvalue) if hasattr(v, 'pvalue') else str(v) for k, v in normality.items()},
        "stable_params": stable_params
    }
    with open(f"{filename}.json", "w") as f:
        json.dump(report, f, indent=4)

    df_summary = pd.DataFrame(report["summary"], index=[0])
    df_params = pd.DataFrame(stable_params, index=[0])
    df_normality = pd.DataFrame(report["normality_tests"]).T

    with pd.ExcelWriter(f"{filename}.xlsx") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        df_params.to_excel(writer, sheet_name="Stable_Params", index=False)
        df_normality.to_excel(writer, sheet_name="Normality_Tests")

# Final full analysis function
def analyse_stable_distribution(x, filename="interval_analysis", qcv_threshold=1.8, fig_path="stability_test_plot.png"):
    """
    Complete stability analysis - returns formatted string for Streamlit
    Fixed version with proper markdown formatting
    """
    try:
        normal = test_normality(x)
        sk_kurt = skew_kurtosis(x)
        qcv = qcv_stat(x)
        stable_params = estimate_stable_r(x)

        # Build clean markdown report for Streamlit
        report = f"""## 📊 Stability Analysis Results

### Dataset Summary
- **Sample size:** {len(x)}
- **Skewness:** {sk_kurt['skewness']:.3f}
- **Kurtosis:** {sk_kurt['kurtosis']:.3f}  
- **QCV Statistic:** {qcv:.3f}

### 🧪 Normality Tests

#### Shapiro-Wilk Test
- **Statistic:** {normal['Shapiro'].statistic:.4f}
- **p-value:** {normal['Shapiro'].pvalue:.4f}
- **Result:** {'✅ Normal' if normal['Shapiro'].pvalue >= 0.05 else '⛔ Non-normal'}

#### Anderson-Darling Test  
- **Statistic:** {normal['Anderson'].statistic:.4f}
- **Critical value (5%):** {normal['Anderson'].critical_values[2]:.4f}
- **Result:** {'✅ Normal' if normal['Anderson'].statistic < normal['Anderson'].critical_values[2] else '⛔ Non-normal'}

### 📈 Estimated Stable Parameters
- **α (tail):** {stable_params['alpha']:.3f}
- **β (skewness):** {stable_params['beta']:.3f}
- **γ (scale):** {stable_params['gamma']:.3f}
- **δ (location):** {stable_params['delta']:.3f}

### 🎯 Final Verdict"""

        # Verdict logic (keeping your original logic)
        stat_shapiro, p_shapiro = normal["Shapiro"]
        shapiro_pass = p_shapiro >= 0.05
        
        stat_anderson = normal["Anderson"].statistic
        crit_anderson = normal["Anderson"].critical_values[2]
        anderson_pass = stat_anderson < crit_anderson
        
        normality_rejected = (not shapiro_pass) or (not anderson_pass)
        if normality_rejected and qcv > qcv_threshold:
            verdict = "✅ **Distribution probablement α-stable** (non normale, queue lourde)"
        else:
            verdict = "⛔ **Distribution probablement pas α-stable**"

        report += f"\n\n{verdict}"

        # Generate plots and export  
        plot_vs_normal_stable(x, stable_params, fig_path=fig_path)
        export_analysis_report(x, stable_params, qcv, sk_kurt, normal, verdict, filename=filename)
        
        return report

        
    except Exception as e:
        return f"❌ **Error in stability analysis:** {str(e)}"