import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from scipy.stats import norm, skew, kurtosis, shapiro, kstest, anderson
from statsmodels.stats.stattools import jarque_bera
from rpy2.robjects import FloatVector
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
from .r_interface import libstable4u,alphastable

# QCV R function setup
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
        return None

def qcv_stat(x):
    x_r = FloatVector(x.tolist())
    return float(qcv_test.qcv_stat(x_r)[0])

def fit_em_mixture_r(x):
    x_r = FloatVector(x.tolist())
    try:
        return alphastable.emstabledist(x_r, 2)
    except Exception as e:
        print(f"EM fitting error: {e}")
        return None

def plot_vs_normal_stable(x, params_stable):
    xx = np.linspace(min(x), max(x), 500)
    norm_pdf = norm.pdf(xx, np.mean(x), np.std(x))
    x_r = FloatVector(xx.tolist())
    pars = FloatVector([params_stable[k] for k in ["alpha", "beta", "gamma", "delta"]])
    stable_pdf = libstable4u.stable_pdf(x_r, pars)
    plt.figure(figsize=(10, 6))
    plt.hist(x, bins=50, density=True, alpha=0.4, label='Data')
    plt.plot(xx, norm_pdf, 'r--', label='Normal PDF')
    plt.plot(xx, stable_pdf, 'b-', label='Stable PDF (R)')
    plt.title("Empirical vs Normal vs Stable PDF")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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
def analyse_stable_distribution(x, filename="interval_analysis", qcv_threshold=1.8):
    print("✅ Analyse de la série...")

    normal = test_normality(x)
    sk_kurt = skew_kurtosis(x)
    qcv = qcv_stat(x)
    stable_params = estimate_stable_r(x)

    print(f"Skewness: {sk_kurt['skewness']:.3f}")
    print(f"Kurtosis: {sk_kurt['kurtosis']:.3f}")
    print(f"QCV: {qcv:.3f}")

    print("\n🧪 Tests de normalité (Shapiro & Anderson uniquement) :")

    # — Analyse du test de Shapiro
    stat_shapiro, p_shapiro = normal["Shapiro"]
    shapiro_pass = p_shapiro >= 0.05
    flag_shapiro = "✅ passe" if shapiro_pass else "⛔ rejette"
    print(f"• Shapiro      → stat={stat_shapiro:.4f}, p={p_shapiro:.4f} → {flag_shapiro}")

    # — Analyse du test de Anderson
    stat_anderson = normal["Anderson"].statistic
    crit_anderson = normal["Anderson"].critical_values[2]  # seuil à 5%
    anderson_pass = stat_anderson < crit_anderson
    flag_anderson = "✅ passe" if anderson_pass else "⛔ rejette"
    print(f"• Anderson     → stat={stat_anderson:.4f}, seuil_5%={crit_anderson:.4f} → {flag_anderson}")

    # Verdict basé sur : au moins 1 test rejette la normalité ET QCV > seuil
    normality_rejected = (not shapiro_pass) or (not anderson_pass)
    if normality_rejected and qcv > qcv_threshold:
        verdict = "✅ Distribution probablement α-stable (non normale, queue lourde)"
    else:
        verdict = "⛔ Distribution probablement pas α-stable"

    print("\n→", verdict)

    plot_vs_normal_stable(x, stable_params)
    export_analysis_report(x, stable_params, qcv, sk_kurt, normal, verdict, filename=filename)
    return verdict



