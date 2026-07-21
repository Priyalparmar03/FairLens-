"""
Validation helpers.

Author: FairLens
"""

from __future__ import annotations

import pandas as pd


def validate_dataframe(df: pd.DataFrame):

    if df.empty:

        raise ValueError(

            "Input dataframe is empty."
        )


def validate_columns(

    df: pd.DataFrame,

    required_columns,

):

    missing = [

        c

        for c in required_columns

        if c not in df.columns

    ]

    if missing:

        raise ValueError(

            f"Missing columns: {missing}"

        )


def validate_binary(series: pd.Series):

    if len(series.unique()) != 2:

        raise ValueError(

            "Expected binary labels."

        )


def validate_sensitive_feature(

    df: pd.DataFrame,

    column: str,

):

    if column not in df.columns:

        raise ValueError(

            f"Sensitive feature '{column}' not found."

        )


def validate_predictions(

    y_true,

    y_pred,

):

    if len(y_true) != len(y_pred):

        raise ValueError(

            "Prediction lengths differ."

        )