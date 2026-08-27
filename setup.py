from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent

setup(
    name="pg-m2tn",
    version="2.0.0",
    description="PG-M2TN August 2026 revision reproduction code",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Shuhao Chen",
    author_email="2023333541008@mails.zstu.edu.cn",
    url="https://github.com/shuhaochen618-svg/PG-M2TN",
    packages=find_packages(),
    python_requires=">=3.11,<3.12",
    install_requires=[
        "torch>=2.5,<2.6",
        "numpy>=2.4,<2.5",
        "tqdm>=4.68,<4.69",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
)
