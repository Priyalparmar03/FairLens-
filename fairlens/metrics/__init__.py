"""
FairLens Metrics Package

Provides a unified interface for fairness metrics
across multiple fairness libraries.
"""

from .base import (
    BaseMetricEngine,
    MetricCollection,
    MetricResult,
)

from .fairlearn import FairlearnMetricEngine
from .aif360 import AIF360MetricEngine

from .registry import (
    METRIC_REGISTRY,
    get_metric,
    list_metrics,
)

__all__ = [
    "BaseMetricEngine",
    "MetricCollection",
    "MetricResult",
    "FairlearnMetricEngine",
    "AIF360MetricEngine",
    "METRIC_REGISTRY",
    "get_metric",
    "list_metrics",
]