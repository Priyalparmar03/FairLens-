"""
Fairlearn Metric Engine

Wrapper around Fairlearn fairness metrics.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equal_opportunity_difference,
)

from .base import (
    BaseMetricEngine,
    MetricCollection,
)

from .registry import (
    list_metrics,
    get_metric,
)

from .utils import (
    build_metric,
)


class FairlearnMetricEngine(BaseMetricEngine):
    """
    Fairlearn metric engine.

    Computes fairness metrics using Fairlearn while
    returning them in FairLens format.
    """

    library_name = "fairlearn"

    def supported_metrics(self) -> List[str]:

        return list_metrics()

    def compute(
        self,
        df: pd.DataFrame,
        y_true: str,
        y_pred: str,
        sensitive_feature: str,
    ) -> MetricCollection:

        self.reset()

        y_t = df[y_true]
        y_p = df[y_pred]
        sensitive = df[sensitive_feature]

        # --------------------------------------------------
        # Demographic Parity Difference
        # --------------------------------------------------

        definition = get_metric(
            "demographic_parity_difference"
        )

        value = demographic_parity_difference(
            y_true=y_t,
            y_pred=y_p,
            sensitive_features=sensitive,
        )

        self.results.add(
            build_metric(
                name=definition.canonical_name,
                value=value,
                threshold=definition.threshold,
                description=definition.description,
            )
        )

        # --------------------------------------------------
        # Demographic Parity Ratio
        # --------------------------------------------------

        definition = get_metric(
            "demographic_parity_ratio"
        )

        value = demographic_parity_ratio(
            y_true=y_t,
            y_pred=y_p,
            sensitive_features=sensitive,
        )

        self.results.add(
            build_metric(
                definition.canonical_name,
                value,
                definition.threshold,
                definition.description,
            )
        )

        # --------------------------------------------------
        # Equalized Odds Difference
        # --------------------------------------------------

        definition = get_metric(
            "equalized_odds_difference"
        )

        value = equalized_odds_difference(
            y_true=y_t,
            y_pred=y_p,
            sensitive_features=sensitive,
        )

        self.results.add(
            build_metric(
                definition.canonical_name,
                value,
                definition.threshold,
                definition.description,
            )
        )

        # --------------------------------------------------
        # Equal Opportunity Difference
        # --------------------------------------------------

        definition = get_metric(
            "equal_opportunity_difference"
        )

        value = equal_opportunity_difference(
            y_true=y_t,
            y_pred=y_p,
            sensitive_features=sensitive,
        )

        self.results.add(
            build_metric(
                definition.canonical_name,
                value,
                definition.threshold,
                definition.description,
            )
        )

        return self.results