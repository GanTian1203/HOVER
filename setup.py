"""Editable-install compatibility for the original Python 3.8 environment."""
from setuptools import find_packages, setup

setup(
    name="hover-reproduction",
    version="0.4.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"hover": ["configuration/*.json"]},
    python_requires=">=3.8",
    install_requires=["torch>=2.0,<2.1", "numpy>=1.23,<1.24", "pandas>=1.5,<1.6", "scipy>=1.10,<1.11", "matplotlib>=3.7,<3.8", "psutil>=5.9,<6"],
    extras_require={"baselines": ["einops==0.5.0", "reformer-pytorch==1.4.4", "timm==0.6.13", "charset-normalizer==3.3.2", "seaborn==0.13.2", "torchvision>=0.15,<0.16"]},
    entry_points={"console_scripts": ["hover=hover.cli:main"]},
)
