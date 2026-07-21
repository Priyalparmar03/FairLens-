"""
Shared utility functions for FairLens metrics.

These utilities are used by all metric engines (Fairlearn, AIF360, etc.)
to validate inputs, safely compute values, normalize outputs, and
format metric results.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from .base import MetricResult


# ==========================================================
# Constants
# ==========================================================

DEFAULT_THRESHOLD = 0.10

EPSILON = 1e-12


# ==========================================================
# Safe Division
# ==========================================================

def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    """
    Safely divide two numbers.

    Returns
    -------
    float
        0 if denominator is zero.
    """

    if abs(denominator) < EPSILON:
        return 0.0

    return numerator / denominator


# ==========================================================
# Threshold Checking
# ==========================================================

def check_threshold(
    value: float,
    threshold: float = DEFAULT_THRESHOLD,
) -> bool:
    """
    Determine whether a fairness metric passes.

    Most fairness metrics are considered acceptable when
    their absolute value is below the threshold.

    Returns
    -------
    bool
    """

    return abs(value) <= threshold


# ==========================================================
# Normalize Metric
# ==========================================================

def normalize_metric(value: float) -> float:
    """
    Convert numpy scalars to Python float.

    Ensures JSON serialization compatibility.
    """

    if isinstance(value, np.generic):
        return float(value)

    return float(value)


# ==========================================================
# Extract Sensitive Groups
# ==========================================================

def extract_groups(
    df: pd.DataFrame,
    sensitive_feature: str,
) -> List[Any]:
    """
    Return unique sensitive groups.

    Example
    -------
    Male
    Female
    """

    if sensitive_feature not in df.columns:
        raise ValueError(
            f"'{sensitive_feature}' not found in dataframe."
        )

    return sorted(df[sensitive_feature].dropna().unique())


# ==========================================================
# Group Counts
# ==========================================================

def group_counts(
    df: pd.DataFrame,
    sensitive_feature: str,
) -> Dict[Any, int]:
    """
    Count samples in each sensitive group.
    """

    return (
        df[sensitive_feature]
        .value_counts()
        .to_dict()
    )


# ==========================================================
# Metric Factory
# ==========================================================

def build_metric(
    name: str,
    value: float,
    threshold: float = DEFAULT_THRESHOLD,
    description: str = "",
) -> MetricResult:
    """
    Construct MetricResult object.
    """

    value = normalize_metric(value)

    return MetricResult(
        name=name,
        value=value,
        threshold=threshold,
        passed=check_threshold(
            value,
            threshold,
        ),
        description=description,
    )


# ==========================================================
# Dictionary Conversion
# ==========================================================

def metrics_to_dict(
    metrics: List[MetricResult],
) -> Dict[str, Dict]:
    """
    Convert MetricResult list into dictionary.
    """

    return {
        metric.name: metric.as_dict()
        for metric in metrics
    }


# ==========================================================
# Validate Columns
# ==========================================================

def validate_columns(
    df: pd.DataFrame,
    columns: List[str],
):
    """
    Ensure required columns exist.
    """

    missing = [
        c for c in columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )


# ==========================================================
# Prediction Validation
# ==========================================================

def validate_predictions(
    y_true,
    y_pred,
):
    """
    Validate prediction arrays.
    """

    if len(y_true) != len(y_pred):

        raise ValueError(
            "Prediction lengths differ."
        )


# ==========================================================
# Binary Validation
# ==========================================================

def validate_binary(
    series: pd.Series,
):
    """
    Verify binary target.

    Raises
    ------
    ValueError
    """

    unique = sorted(series.unique())

    if len(unique) != 2:
        raise ValueError(
            "Binary labels expected."
        )


# ==========================================================
# Pretty Formatting
# ==========================================================

def format_percentage(
    value: float,
    digits: int = 2,
) -> str:
    """
    Format decimal as percentage.
    """

    return f"{100*value:.{digits}f}%"