"""
Base interfaces and dataclasses for FairLens metrics.

Every metric implementation (Fairlearn, AIF360, or future libraries)
must inherit from BaseMetricEngine.

Author: FairLens
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd


# ==========================================================
# Metric Result
# ==========================================================

@dataclass(slots=True)
class MetricResult:
    """
    Represents a single fairness metric.

    Parameters
    ----------
    name
        Human-readable metric name.

    value
        Numerical value.

    threshold
        Pass/fail threshold.

    passed
        Whether metric satisfies threshold.

    description
        Optional explanation.
    """

    name: str
    value: float
    threshold: Optional[float] = None
    passed: Optional[bool] = None
    description: str = ""

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""

        return {
            "name": self.name,
            "value": self.value,
            "threshold": self.threshold,
            "passed": self.passed,
            "description": self.description,
        }


# ==========================================================
# Metric Collection
# ==========================================================

@dataclass(slots=True)
class MetricCollection:
    """
    Stores multiple MetricResult objects.
    """

    library: str

    metrics: Dict[str, MetricResult] = field(default_factory=dict)

    def add(self, metric: MetricResult) -> None:
        """Add metric."""

        self.metrics[metric.name] = metric

    def get(self, name: str) -> MetricResult:

        return self.metrics[name]

    def to_dict(self) -> Dict[str, Any]:

        return {
            name: metric.as_dict()
            for name, metric in self.metrics.items()
        }

    def __len__(self):

        return len(self.metrics)

    def __iter__(self):

        return iter(self.metrics.values())


# ==========================================================
# Base Metric Engine
# ==========================================================

class BaseMetricEngine(ABC):
    """
    Abstract metric engine.

    Every fairness library wrapper must implement this class.

    Examples
    --------
    FairlearnMetricEngine

    AIF360MetricEngine
    """

    library_name = "base"

    def __init__(self):

        self.results = MetricCollection(
            library=self.library_name
        )

    @abstractmethod
    def compute(
        self,
        df: pd.DataFrame,
        y_true: str,
        y_pred: str,
        sensitive_feature: str,
    ) -> MetricCollection:
        """
        Compute all supported fairness metrics.
        """
        raise NotImplementedError

    @abstractmethod
    def supported_metrics(self):
        """
        Return supported metric names.
        """
        raise NotImplementedError

    def reset(self):

        self.results = MetricCollection(
            library=self.library_name
        )

    def __repr__(self):

        return (
            f"{self.__class__.__name__}"
            f"(library='{self.library_name}')"
        )