
---

````markdown
# 🌌 Alpha-Stable Mixture Estimation Package

A comprehensive Python package for simulating, estimating, and visualizing **alpha-stable mixture distributions**. This toolkit is built for statisticians, data scientists, and researchers who work with heavy-tailed or skewed data models where Gaussian assumptions fall short.

---

## ✨ Key Features

🔍 **Robust Estimation Methods**
- Empirical Characteristic Function (ECF) estimators: kernel-based, weighted OLS
- Maximum Likelihood Estimation (MLE)
- Quantile-based and CDF-based methods

🧠 **Mixture Modeling with EM**
- Expectation-Maximization (EM) algorithm for two-component alpha-stable mixtures
- Customizable estimators within the EM loop
- Optional integration with **Gibbs sampling** for Bayesian refinement

📈 **Visualization & Simulation**
- Tools to generate synthetic data from alpha-stable distributions
- Interactive visualization with **Streamlit dashboard**

🔗 **R Integration (via `rpy2`)**
- Seamless use of R's `stabledist` package to evaluate stable densities and CDFs
- Benefit from mature statistical tools while staying in Python

🧪 **Testing & Evaluation**
- Pre-built evaluation metrics and test datasets
- Model diagnostics and goodness-of-fit

---

## 🚀 Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/yourname/alpha_stable_mixture.git
cd alpha_stable_mixture
pip install -e .
````

Install required R package:

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
from alpha_stable_mixture import generate_sample, em

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

* `numpy`, `scipy`, `matplotlib`, `seaborn`
* `rpy2`, `pandas`, `tqdm`

---

## 📁 Project Structure

```
alpha_stable_mixture/
│
├── alpha_stable_mixture/     # Core modules
│   ├── em.py, gibbs.py, ecf.py, etc.
│   └── interface/            # Streamlit dashboard & data preprocessor
│
├── tests/                    # Test scripts
├── README.md
├── setup.py
├── pyproject.toml
└── requirements.txt
```

---

## 👨‍💻 Author

**Adam Najib**
Email: \[[najibadam145@gmail.com](mailto:najibadam145@gmail.com)]
GitHub: [@AdamNajib](https://github.com/AdamNajib)

---


