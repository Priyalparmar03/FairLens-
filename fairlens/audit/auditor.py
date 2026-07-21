from __future__ import annotations

from typing import Dict

import pandas as pd

from .validator import AuditValidator
from .pipeline import AuditPipeline


class FairnessAuditor:
    """
    Main entry point of FairLens.

    Coordinates

    - validation
    - fairness metrics
    - MAI
    - EU AI Act classification
    - report generation
    """

    def __init__(self):

        self.pipeline = AuditPipeline()

    def audit(
        self,
        df: pd.DataFrame,
        y_true: str,
        y_pred: str,
        sensitive_feature: str,
    ) -> Dict:
        """
        Run complete fairness audit.

        Parameters
        ----------
        df
            Dataset.

        y_true
            Name of ground truth column.

        y_pred
            Prediction column.

        sensitive_feature
            Protected attribute.

        Returns
        -------
        dict
            Complete audit results.
        """

        AuditValidator.validate_dataframe(df)

        AuditValidator.validate_columns(
            df,
            (
                y_true,
                y_pred,
                sensitive_feature,
            ),
        )

        AuditValidator.validate_prediction_lengths(
            df[y_true],
            df[y_pred],
        )

        context = {
            "data": df,
            "y_true": y_true,
            "y_pred": y_pred,
            "sensitive": sensitive_feature,
        }

        result = self.pipeline.run(context)

        return result