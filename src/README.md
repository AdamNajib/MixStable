
# 🌌 MixStable — Alpha-Stable Mixture Estimation Package

[![PyPI Version](https://img.shields.io/pypi/v/alpha-stable-mixture.svg)](https://pypi.org/project/alpha-stable-mixture/)
[![Python Versions](https://img.shields.io/pypi/pyversions/alpha-stable-mixture.svg)](https://pypi.org/project/alpha-stable-mixture/)
[![License](https://img.shields.io/pypi/l/alpha-stable-mixture.svg)](https://pypi.org/project/alpha-stable-mixture/)

A comprehensive Python package for **simulating**, **estimating**, and **visualizing** alpha-stable mixture distributions.  
This toolkit is designed for statisticians, data scientists, and researchers working with **heavy-tailed** or **skewed data**, where Gaussian models fail to capture real-world complexity.

---

## ✨ Key Features

🔍 **Robust Estimation Methods**
- Empirical Characteristic Function (ECF) estimators: kernel-based and weighted OLS  
- Maximum Likelihood Estimation (MLE)  
- Quantile-based and CDF-based approaches  

🧠 **Mixture Modeling with EM**
- EM algorithm for two-component alpha-stable mixtures  
- Flexible estimator choice inside the EM loop  
- Optional integration with **Gibbs sampling** for Bayesian refinement  

📈 **Visualization & Simulation**
- Generate synthetic alpha-stable data  
- Built-in visualization tools and an **interactive Streamlit dashboard**  

🔗 **R Integration (via `rpy2`)**
- Uses R’s `stabledist` package to evaluate stable densities and CDFs  
- Leverage mature R statistical methods directly from Python  

🧪 **Testing & Evaluation**
- Built-in diagnostics and model-fit tools  
- Evaluation metrics and example datasets  

---

## 🚀 Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/yourname/alpha_stable_mixture.git
cd alpha_stable_mixture
pip install -e .
````

Install the required R package (for rpy2 integration):

```r
install.packages("stabledist")
```

---

## 🧪 Running the Interactive Dashboard

To launch the Streamlit app for interactive parameter tuning and visualization:

```bash
streamlit run interface/app.py
```

---

## 🛠️ Usage Example

Estimate parameters from synthetic data:

```python
from mixstable import generate_sample, em

# Generate alpha-stable samples
samples = generate_sample.generate_alpha_stable(alpha=1.7, beta=0, gamma=1, delta=0, size=1000)

# Fit a two-component mixture model using EM
result = em.run_em_algorithm(samples, num_components=2, max_iter=50)
print(result['params'])
```

---

## 📦 Requirements

* Python ≥ 3.8
* R (if using `r_interface`)
* `stabledist` R package

Python dependencies (auto-installed from `requirements.txt`):

* `numpy`, `scipy`, `matplotlib`, `seaborn` , `statsmodels`
* `rpy2`, `pandas`, `tqdm` , `scikit-learn` , `streamlit`

---

## 📁 Project Structure

```
alpha_stable_mixture/
│
├── src/mixstable/    # Core modules
│   ├── em.py, gibbs.py, ecf.py, ...
│   └── interface/               # Streamlit dashboard
│
├── tests/                       # Test scripts
├── README.md
├── setup.py
├── pyproject.toml
└── requirements.txt

```
## 🧩 Package Overview

| Module               | Description                                       |
| :------------------- | :------------------------------------------------ |
| `em.py`              | Expectation-Maximization algorithm implementation |
| `gibbs.py`           | Gibbs sampling routines for Bayesian inference    |
| `ecf.py`             | Empirical characteristic function estimators      |
| `generate_sample.py` | Alpha-stable data generation tools                |
| `interface/`         | Streamlit dashboard and helper scripts            |
| `tests/`             | Unit and integration tests                        |

---
## 📊 Example Output

Example EM fit visualization (from Streamlit app):

Iteration 50/50
Component 1: α = 1.72, β = 0.05, γ = 1.1, δ = 0.02
Component 2: α = 1.54, β = -0.10, γ = 0.8, δ = 2.4
Log-likelihood: -1856.42

---

## 🧠 Theoretical Background

Alpha-stable distributions generalize the Gaussian family to model heavy-tailed and skewed data.
They are defined by four parameters:

| Parameter | Symbol | Meaning                             |
| :-------- | :----- | :---------------------------------- |
| Stability | α      | Controls tail thickness (0 < α ≤ 2) |
| Skewness  | β      | Controls asymmetry (−1 ≤ β ≤ 1)     |
| Scale     | γ      | Controls spread (> 0)               |
| Location  | δ      | Controls center (real number)       |


This package provides both simulation tools and inference routines for mixtures of alpha-stable laws, extending beyond Gaussian mixtures.

---

## 📚 References

Hajjaji, O., Manou-Abi, S. M., & Slaoui, Y. (2024).
Parameter estimation for stable distributions and their mixture.
Journal of Applied Statistics, 52(8), 1594–1627.
https://doi.org/10.1080/02664763.2024.2434627

---

## 👨‍💻 Authors

**Solym Manou-Abi**
Email: \[[solym.manou.abi@univ-poitiers.fr](mailto:solym.manou.abi@univ-poitiers.fr)]

**Adam Najib**
Email: \[[najibadam145@gmail.com](mailto:najibadam145@gmail.com)]
GitHub: [@AdamNajib](https://github.com/AdamNajib)

**Yousri Slaoui**
Email: \[[Yousri.Slaoui@math.univ-poitiers.fr](mailto:Yousri.Slaoui@math.univ-poitiers.fr)]

---

