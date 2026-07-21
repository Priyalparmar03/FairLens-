
"""
fairlens.datasets.adult

Adult Income dataset loader for FairLens.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from .preprocessing import prepare_dataset

logger = logging.getLogger(__name__)

ADULT_COLUMNS = [
    "age","workclass","fnlwgt","education","education_num",
    "marital_status","occupation","relationship","race","sex",
    "capital_gain","capital_loss","hours_per_week",
    "native_country","income",
]


def load_adult_dataset(
    path: str | Path,
    *,
    target_column: str = "income",
    sensitive_columns: Optional[Sequence[str]] = ("sex", "race"),
    prediction_column: str = "prediction",
    preprocess: bool = True,
) -> pd.DataFrame:
    """
    Load the UCI Adult Income dataset.

    Parameters
    ----------
    path:
        Path to ``adult.data``.
    preprocess:
        If True, returns a dataframe with a prediction column generated
        by a baseline Logistic Regression model.

    Returns
    -------
    pandas.DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    logger.info("Loading Adult dataset from %s", path)

    df = pd.read_csv(
        path,
        header=None,
        names=ADULT_COLUMNS,
        skipinitialspace=True,
        na_values=["?", " ?"],
    )

    df.dropna(inplace=True)

    df[target_column] = (
        df[target_column]
        .astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)
    )

    if not preprocess:
        return df

    processed_df, model, accuracy = prepare_dataset(
        df=df,
        target_column=target_column,
        sensitive_columns=sensitive_columns,
        prediction_column=prediction_column,
    )

    processed_df.attrs["model"] = model
    processed_df.attrs["baseline_accuracy"] = accuracy

    logger.info(
        "Adult dataset prepared successfully. Baseline accuracy = %.4f",
        accuracy,
    )

    return processed_df


__all__ = [
    "ADULT_COLUMNS",
    "load_adult_dataset",
]
