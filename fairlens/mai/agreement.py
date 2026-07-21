"""
Metric Agreement Index (MAI)

Novel contribution of FairLens.

MAI measures how consistently multiple fairness
libraries agree with one another.
"""

from __future__ import annotations

from typing import Dict

from .weighting import MetricWeighting


class MetricAgreementIndex:
    """
    Compute agreement between Fairlearn
    and AIF360.

    Agreement Score

        1.0 = Perfect Agreement

        0.0 = Complete Disagreement
    """

    def __init__(
        self,
        weighting: MetricWeighting | None = None,
    ):

        self.weighting = (
            weighting
            if weighting
            else MetricWeighting()
        )

    def compute(
        self,
        fairlearn_results: Dict,
        aif360_results: Dict,
    ) -> Dict:
        """
        Compute MAI.
        """

        agreements = {}

        weighted_sum = 0.0

        total_weight = 0.0

        for metric in fairlearn_results:

            if metric not in aif360_results:
                continue

            fl = fairlearn_results[metric]["value"]

            ai = aif360_results[metric]["value"]

            weight = self.weighting.get(metric)

            # ---------------------------------
            # Agreement Formula
            # ---------------------------------

            score = 1.0 - abs(fl - ai)

            score = max(
                0.0,
                min(
                    1.0,
                    score,
                ),
            )

            agreements[metric] = score

            weighted_sum += score * weight

            total_weight += weight

        overall = (
            weighted_sum / total_weight
            if total_weight
            else 0.0
        )

        return {
            "overall_agreement": round(
                overall,
                4,
            ),
            "metric_agreements": agreements,
        }

    def classify(
        self,
        score: float,
    ) -> str:
        """
        Human-readable interpretation.
        """

        if score >= 0.95:
            return "Excellent"

        if score >= 0.85:
            return "High"

        if score >= 0.70:
            return "Moderate"

        if score >= 0.50:
            return "Low"

        return "Very Low"