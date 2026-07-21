from __future__ import annotations

from typing import Iterable

import pandas as pd


class AuditValidator:
    """
    Validates all user inputs before an audit begins.
    """

    REQUIRED_COLUMNS = (
        "y_true",
        "y_pred",
        "sensitive_feature",
    )

    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> None:
        """
        Ensure dataframe is valid.

        Raises
        ------
        ValueError
            If dataframe is empty.
        """

        if df.empty:
            raise ValueError("Input dataframe is empty.")

    @staticmethod
    def validate_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
        """
        Ensure required columns exist.
        """

        missing = [c for c in columns if c not in df.columns]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

    @staticmethod
    def validate_prediction_lengths(
        y_true,
        y_pred,
    ) -> None:
        """
        Validate prediction lengths.
        """

        if len(y_true) != len(y_pred):
            raise ValueError(
                "y_true and y_pred have different lengths."
            )

    @staticmethod
    def validate_sensitive_feature(feature) -> None:
        """
        Ensure sensitive feature exists.
        """

        if feature is None:
            raise ValueError(
                "Sensitive feature cannot be None."
            )