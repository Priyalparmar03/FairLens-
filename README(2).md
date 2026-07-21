# FairLens

> **FairLens** is an end-to-end AI fairness auditing toolkit for
> evaluating machine learning models using **Fairlearn**, **AIF360**, a
> **Metric Agreement Index (MAI)**, and **EU AI Act Annex III** risk
> classification.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Beta-orange)

## Overview

FairLens provides a unified framework for auditing classification models
for algorithmic fairness. It combines multiple fairness libraries,
generates human-readable reports, and maps results to EU AI Act risk
categories.

### Key Features

-   Fairlearn fairness metrics
-   AIF360 fairness metrics
-   Metric Agreement Index (MAI)
-   EU AI Act Annex III risk classification
-   Markdown, HTML, and JSON reports
-   Built-in dataset loaders
-   Synthetic dataset generator
-   Extensible audit pipeline
-   Configurable thresholds
-   PyPI-friendly package structure

------------------------------------------------------------------------

# Architecture

``` text
Dataset
   │
   ▼
Validation
   │
   ▼
Preprocessing
   │
   ▼
Fairlearn Metrics
   │
   ▼
AIF360 Metrics
   │
   ▼
Metric Agreement Index
   │
   ▼
Annex III Classification
   │
   ▼
Report Generation
```

------------------------------------------------------------------------

# Installation

``` bash
git clone https://github.com/yourusername/fairlens.git
cd fairlens
pip install -e .
```

or

``` bash
pip install fairlens
```

------------------------------------------------------------------------

# Supported Datasets

  Dataset         Domain             Sensitive Attributes
  --------------- ------------------ --------------------------
  Adult           Income             Sex, Race
  COMPAS          Criminal Justice   Race, Sex
  German Credit   Lending            Age, Sex
  Synthetic       Demo / Testing     Gender, Race, Disability

------------------------------------------------------------------------

# Quick Start

``` python
from fairlens.datasets import load_dataset
from fairlens.audit import FairnessAuditor

df = load_dataset(
    "adult",
    path="data/adult/adult.data"
)

auditor = FairnessAuditor()

result = auditor.audit(
    df=df,
    y_true="income",
    y_pred="prediction",
    sensitive_feature="sex"
)

print(result)
```

------------------------------------------------------------------------

# Datasets API

``` python
from fairlens.datasets import load_dataset

adult = load_dataset("adult", path="data/adult/adult.data")
compas = load_dataset("compas", path="data/compas/compas-scores-two-years.csv")
german = load_dataset("german", path="data/german/german.data")
synthetic = load_dataset("synthetic", n_samples=10000)
```

------------------------------------------------------------------------

# Reports

FairLens can generate:

-   Markdown reports
-   HTML reports
-   JSON reports

------------------------------------------------------------------------

# Package Structure

``` text
fairlens/
│
├── audit/
├── annex3/
├── config/
├── datasets/
├── mai/
├── metrics/
├── report/
├── schemas/
├── utils/
└── tests/
```

------------------------------------------------------------------------

# Fairness Metrics

Examples include:

-   Demographic Parity Difference
-   Demographic Parity Ratio
-   Equal Opportunity Difference
-   Equalized Odds Difference
-   Disparate Impact
-   Statistical Parity Difference

------------------------------------------------------------------------

# Metric Agreement Index (MAI)

MAI measures agreement between Fairlearn and AIF360 metrics to increase
confidence in fairness assessments.

------------------------------------------------------------------------

# EU AI Act

FairLens supports Annex III mapping for identifying High-Risk AI
systems.

Typical domains include:

-   Employment
-   Education
-   Credit
-   Law Enforcement
-   Migration
-   Essential Public Services

------------------------------------------------------------------------

# Testing

``` bash
pytest
```

------------------------------------------------------------------------

# Roadmap

-   Plugin architecture
-   More datasets
-   Regression fairness
-   Explainability integration
-   Dashboard
-   CI/CD automation

------------------------------------------------------------------------

# Contributing

1.  Fork the repository.
2.  Create a feature branch.
3.  Commit your changes.
4.  Open a Pull Request.

------------------------------------------------------------------------

# Citation

``` bibtex
@software{fairlens,
  title={FairLens: A Unified AI Fairness Auditing Toolkit},
  author={Your Name},
  year={2026}
}
```

------------------------------------------------------------------------

# License

MIT License

------------------------------------------------------------------------

# Acknowledgements

-   Fairlearn
-   IBM AIF360
-   Scikit-learn
-   NumPy
-   Pandas
-   EU AI Act
