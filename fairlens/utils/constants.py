"""
Global constants used across FairLens.

Author: FairLens
"""

from __future__ import annotations

from pathlib import Path

# ==========================================================
# Package Information
# ==========================================================

PACKAGE_NAME = "fairlens"

VERSION = "0.1.0"

AUTHOR = "FairLens"

# ==========================================================
# Default Thresholds
# ==========================================================

DEFAULT_FAIRNESS_THRESHOLD = 0.10

DEFAULT_DISPARATE_IMPACT_THRESHOLD = 0.80

# ==========================================================
# Supported Libraries
# ==========================================================

SUPPORTED_METRIC_LIBRARIES = [

    "fairlearn",

    "aif360",
]

# ==========================================================
# Supported Reports
# ==========================================================

SUPPORTED_REPORT_FORMATS = [

    "markdown",

    "html",

    "json",
]

# ==========================================================
# Dataset Names
# ==========================================================

SUPPORTED_DATASETS = [

    "adult",

    "compas",

    "german",

    "synthetic",
]

# ==========================================================
# Logging
# ==========================================================

DEFAULT_LOG_LEVEL = "INFO"

# ==========================================================
# Directories
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

REPORT_DIR = ROOT_DIR / "reports"

LOG_DIR = ROOT_DIR / "logs"

DATA_DIR = ROOT_DIR / "data"

REPORT_DIR.mkdir(exist_ok=True)

LOG_DIR.mkdir(exist_ok=True)