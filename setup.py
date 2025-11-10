from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="Mixstable",
    version="0.1.7",
    description="Parameter estimation tools for one-dimensional stable distributions and their mixtures.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Solym Manou-Abi, Adam Najib, Yousri Slaoui",
    url="https://github.com/AdamNajib/MixStable",
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
