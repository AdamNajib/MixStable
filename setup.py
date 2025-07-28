from setuptools import setup, find_packages

setup(
    name="alpha_stable_mixture",
    version="0.1.0",
    description="Tools for alpha-stable mixture model estimation and simulation",
    author="Adam Najib",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "streamlit",
        "statsmodels",
        "rpy2"
    ],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
