
"""
fairlens.datasets.compas

COMPAS dataset loader for FairLens.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from .preprocessing import prepare_dataset

logger = logging.getLogger(__name__)

DEFAULT_COLUMNS = [
    "sex",
    "race",
    "age",
    "age_cat",
    "priors_count",
    "juv_fel_count",
    "juv_misd_count",
    "juv_other_count",
    "c_charge_degree",
    "score_text",
    "two_year_recid",
]


def load_compas_dataset(
    path: str | Path,
    *,
    target_column: str = "two_year_recid",
    sensitive_columns: Optional[Sequence[str]] = ("race", "sex"),
    prediction_column: str = "prediction",
    preprocess: bool = True,
) -> pd.DataFrame:
    """
    Load and prepare the ProPublica COMPAS dataset.

    Parameters
    ----------
    path:
        Path to ``compas-scores-two-years.csv``.
    preprocess:
        If True, preprocesses the dataset and generates predictions.

    Returns
    -------
    pandas.DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    logger.info("Loading COMPAS dataset from %s", path)

    df = pd.read_csv(path)

    missing = [c for c in DEFAULT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[DEFAULT_COLUMNS].copy()

    # Basic cleaning
    df.dropna(inplace=True)
    df["race"] = df["race"].astype(str).str.strip()
    df["sex"] = df["sex"].astype(str).str.strip()
    df["score_text"] = df["score_text"].astype(str).str.strip()

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
        "COMPAS dataset prepared successfully. Baseline accuracy = %.4f",
        accuracy,
    )

    return processed_df


__all__ = [
    "DEFAULT_COLUMNS",
    "load_compas_dataset",
]
