"""
AIF360 Metric Engine

Wrapper around IBM AIF360 fairness metrics.

Author: FairLens
"""

from __future__ import annotations

from typing import List

import pandas as pd

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

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


class AIF360MetricEngine(BaseMetricEngine):
    """
    Wrapper around AIF360 fairness metrics.

    Produces the same output format as FairlearnMetricEngine.
    """

    library_name = "aif360"

    def supported_metrics(self) -> List[str]:

        return list_metrics()

    def _build_dataset(
        self,
        df: pd.DataFrame,
        label_column: str,
        protected_column: str,
    ) -> BinaryLabelDataset:
        """
        Convert dataframe into AIF360 BinaryLabelDataset.
        """

        return BinaryLabelDataset(
            favorable_label=1,
            unfavorable_label=0,
            df=df,
            label_names=[label_column],
            protected_attribute_names=[protected_column],
        )

    def compute(
        self,
        df: pd.DataFrame,
        y_true: str,
        y_pred: str,
        sensitive_feature: str,
    ) -> MetricCollection:
        """
        Compute fairness metrics using AIF360.
        """

        self.reset()

        # ---------------------------------------------
        # Ground truth dataset
        # ---------------------------------------------

        truth_df = df.copy()

        truth_dataset = self._build_dataset(
            truth_df,
            y_true,
            sensitive_feature,
        )

        # ---------------------------------------------
        # Prediction dataset
        # ---------------------------------------------

        pred_df = df.copy()

        pred_df[y_true] = pred_df[y_pred]

        pred_dataset = self._build_dataset(
            pred_df,
            y_true,
            sensitive_feature,
        )

        # ---------------------------------------------
        # Determine protected groups
        # ---------------------------------------------

        values = sorted(df[sensitive_feature].unique())

        if len(values) != 2:
            raise ValueError(
                "AIF360 currently expects exactly two protected groups."
            )

        privileged = [
            {sensitive_feature: values[1]}
        ]

        unprivileged = [
            {sensitive_feature: values[0]}
        ]

        metric = ClassificationMetric(
            truth_dataset,
            pred_dataset,
            privileged_groups=privileged,
            unprivileged_groups=unprivileged,
        )

        # ====================================================
        # Statistical Parity Difference
        # ====================================================

        definition = get_metric(
            "demographic_parity_difference"
        )

        self.results.add(
            build_metric(
                name=definition.canonical_name,
                value=metric.statistical_parity_difference(),
                threshold=definition.threshold,
                description=definition.description,
            )
        )

        # ====================================================
        # Disparate Impact
        # ====================================================

        definition = get_metric(
            "demographic_parity_ratio"
        )

        self.results.add(
            build_metric(
                name=definition.canonical_name,
                value=metric.disparate_impact(),
                threshold=definition.threshold,
                description=definition.description,
            )
        )

        # ====================================================
        # Average Odds Difference
        # ====================================================

        definition = get_metric(
            "equalized_odds_difference"
        )

        self.results.add(
            build_metric(
                name=definition.canonical_name,
                value=metric.average_odds_difference(),
                threshold=definition.threshold,
                description=definition.description,
            )
        )

        # ====================================================
        # Equal Opportunity Difference
        # ====================================================

        definition = get_metric(
            "equal_opportunity_difference"
        )

        self.results.add(
            build_metric(
                name=definition.canonical_name,
                value=metric.equal_opportunity_difference(),
                threshold=definition.threshold,
                description=definition.description,
            )
        )

        return self.results