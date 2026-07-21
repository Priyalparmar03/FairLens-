"""
Canonical Fairness Metric Registry.

This module provides a unified registry of fairness metrics
across multiple fairness libraries.

Currently Supported
-------------------
- Fairlearn
- AIF360

Future
------
- AIX360
- FairBench
- Responsible AI Toolbox
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


# ==========================================================
# Metric Definition
# ==========================================================

@dataclass(frozen=True)
class MetricDefinition:
    """
    Canonical fairness metric.

    Parameters
    ----------
    canonical_name
        Standard FairLens metric name.

    fairlearn_name
        Equivalent Fairlearn metric.

    aif360_name
        Equivalent AIF360 metric.

    threshold
        Recommended fairness threshold.

    description
        Human-readable description.
    """

    canonical_name: str
    fairlearn_name: Optional[str]
    aif360_name: Optional[str]
    threshold: float
    description: str


# ==========================================================
# Canonical Registry
# ==========================================================

METRIC_REGISTRY: Dict[str, MetricDefinition] = {

    "demographic_parity_difference": MetricDefinition(
        canonical_name="demographic_parity_difference",
        fairlearn_name="demographic_parity_difference",
        aif360_name="statistical_parity_difference",
        threshold=0.10,
        description=(
            "Difference in positive prediction rate "
            "between protected groups."
        ),
    ),

    "demographic_parity_ratio": MetricDefinition(
        canonical_name="demographic_parity_ratio",
        fairlearn_name="demographic_parity_ratio",
        aif360_name="disparate_impact",
        threshold=0.80,
        description=(
            "Ratio of positive prediction rates "
            "between protected groups."
        ),
    ),

    "equalized_odds_difference": MetricDefinition(
        canonical_name="equalized_odds_difference",
        fairlearn_name="equalized_odds_difference",
        aif360_name="average_odds_difference",
        threshold=0.10,
        description=(
            "Difference in true/false positive rates "
            "between groups."
        ),
    ),

    "equal_opportunity_difference": MetricDefinition(
        canonical_name="equal_opportunity_difference",
        fairlearn_name="equal_opportunity_difference",
        aif360_name="equal_opportunity_difference",
        threshold=0.10,
        description=(
            "Difference in true positive rate "
            "between protected groups."
        ),
    ),
}


# ==========================================================
# Registry Helper Functions
# ==========================================================

def get_metric(name: str) -> MetricDefinition:
    """
    Retrieve a canonical metric.

    Parameters
    ----------
    name
        Canonical FairLens metric name.

    Returns
    -------
    MetricDefinition
    """

    if name not in METRIC_REGISTRY:
        raise KeyError(
            f"Unknown fairness metric: '{name}'"
        )

    return METRIC_REGISTRY[name]


def list_metrics() -> List[str]:
    """
    Return all canonical metrics.
    """

    return sorted(METRIC_REGISTRY.keys())


def list_definitions() -> List[MetricDefinition]:
    """
    Return all metric definitions.
    """

    return list(METRIC_REGISTRY.values())


# ==========================================================
# Library Mapping
# ==========================================================

def fairlearn_mapping() -> Dict[str, str]:
    """
    Mapping:
    Canonical -> Fairlearn
    """

    return {
        metric.canonical_name: metric.fairlearn_name
        for metric in METRIC_REGISTRY.values()
        if metric.fairlearn_name
    }


def aif360_mapping() -> Dict[str, str]:
    """
    Mapping:
    Canonical -> AIF360
    """

    return {
        metric.canonical_name: metric.aif360_name
        for metric in METRIC_REGISTRY.values()
        if metric.aif360_name
    }


# ==========================================================
# Reverse Mapping
# ==========================================================

def reverse_fairlearn_mapping() -> Dict[str, str]:
    """
    Mapping:
    Fairlearn -> Canonical
    """

    return {
        metric.fairlearn_name: metric.canonical_name
        for metric in METRIC_REGISTRY.values()
        if metric.fairlearn_name
    }


def reverse_aif360_mapping() -> Dict[str, str]:
    """
    Mapping:
    AIF360 -> Canonical
    """

    return {
        metric.aif360_name: metric.canonical_name
        for metric in METRIC_REGISTRY.values()
        if metric.aif360_name
    }


# ==========================================================
# Threshold Lookup
# ==========================================================

def get_threshold(metric_name: str) -> float:
    """
    Return fairness threshold.
    """

    return get_metric(metric_name).threshold


# ==========================================================
# Description Lookup
# ==========================================================

def get_description(metric_name: str) -> str:
    """
    Return metric description.
    """

    return get_metric(metric_name).description