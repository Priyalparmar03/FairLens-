"""
Weighting strategies for Metric Agreement Index (MAI).

This module allows different fairness metrics to contribute
with configurable importance.
"""

from __future__ import annotations

from typing import Dict


# ==========================================================
# Default Weights
# ==========================================================

DEFAULT_WEIGHTS = {
    "demographic_parity_difference": 1.0,
    "demographic_parity_ratio": 1.0,
    "equalized_odds_difference": 1.2,
    "equal_opportunity_difference": 1.2,
}


class MetricWeighting:
    """
    Stores weights assigned to fairness metrics.
    """

    def __init__(
        self,
        weights: Dict[str, float] | None = None,
    ):

        self.weights = (
            weights.copy()
            if weights
            else DEFAULT_WEIGHTS.copy()
        )

    def get(self, metric: str) -> float:
        """
        Return weight of a metric.
        """

        return self.weights.get(metric, 1.0)

    def set(
        self,
        metric: str,
        weight: float,
    ) -> None:
        """
        Update metric weight.
        """

        if weight <= 0:
            raise ValueError(
                "Weight must be positive."
            )

        self.weights[metric] = weight

    def all(self) -> Dict[str, float]:

        return self.weights.copy()