"""
fairlens.datasets.german

German Credit dataset loader for FairLens.

Supports:
    • Loading the UCI Statlog German Credit dataset
    • Automatic preprocessing
    • Baseline model training
    • Prediction generation
    • Integration with FairnessAuditor
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from .preprocessing import prepare_dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Official UCI German Credit Column Names
# ---------------------------------------------------------------------

GERMAN_COLUMNS = [
    "Status",
    "Duration",
    "CreditHistory",
    "Purpose",
    "CreditAmount",
    "Savings",
    "EmploymentSince",
    "InstallmentRate",
    "PersonalStatusSex",
    "OtherDebtors",
    "ResidenceSince",
    "Property",
    "Age",
    "OtherInstallmentPlans",
    "Housing",
    "ExistingCredits",
    "Job",
    "PeopleLiable",
    "Telephone",
    "ForeignWorker",
    "Target",
]

# ---------------------------------------------------------------------
# Default Features
# ---------------------------------------------------------------------

DEFAULT_SENSITIVE_COLUMNS = (
    "Age",
    "PersonalStatusSex",
)

TARGET_COLUMN = "Target"

PREDICTION_COLUMN = "prediction"


# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------

def _validate_file(path: Path) -> None:
    """
    Validate that the dataset exists.

    Raises
    ------
    FileNotFoundError
    """

    if not path.exists():
        raise FileNotFoundError(
            f"German Credit dataset not found: {path}"
        )


def _load_raw_dataset(path: Path) -> pd.DataFrame:
    """
    Load the original UCI german.data file.

    Parameters
    ----------
    path:
        Path to german.data

    Returns
    -------
    pandas.DataFrame
    """

    logger.info("Loading German Credit dataset...")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=GERMAN_COLUMNS,
    )

    logger.info("Rows: %d", len(df))

    return df


def _clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning.
    """

    df = df.copy()

    df.drop_duplicates(inplace=True)

    df.dropna(inplace=True)

    return df


# ---------------------------------------------------------------------
# Public Loader
# ---------------------------------------------------------------------

def load_german_dataset(
    path: str | Path,
    *,
    target_column: str = TARGET_COLUMN,
    sensitive_columns: Optional[Sequence[str]] = DEFAULT_SENSITIVE_COLUMNS,
    prediction_column: str = PREDICTION_COLUMN,
    preprocess: bool = True,
) -> pd.DataFrame:
    """
    Load and preprocess the German Credit dataset.

    Parameters
    ----------
    path : str or Path
        Path to ``german.data``.

    target_column : str, default="Target"
        Name of the target column.

    sensitive_columns : Sequence[str], optional
        Sensitive attributes used for fairness auditing.

    prediction_column : str
        Name of prediction column.

    preprocess : bool
        If True, prepares the dataset using FairLens preprocessing.

    Returns
    -------
    pandas.DataFrame
        Ready-to-use dataframe for FairnessAuditor.
    """

    path = Path(path)

    _validate_file(path)

    df = _load_raw_dataset(path)

    df = _clean_dataset(df)

    logger.info("Cleaning completed.")

    # ------------------------------------------------------------
    # Convert target labels
    # ------------------------------------------------------------
    #
    # Original dataset:
    #
    # 1 = Good Credit
    # 2 = Bad Credit
    #
    # Convert to binary:
    #
    # Good -> 1
    # Bad  -> 0
    #
    # ------------------------------------------------------------

    df[target_column] = (
        df[target_column]
        .replace(
            {
                1: 1,
                2: 0,
                "1": 1,
                "2": 0,
            }
        )
        .astype(int)
    )

    logger.info(
        "Target distribution:\n%s",
        df[target_column].value_counts(),
    )

    # ------------------------------------------------------------
    # Ensure sensitive columns exist
    # ------------------------------------------------------------

    if sensitive_columns:

        missing = [
            column
            for column in sensitive_columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing sensitive columns: {missing}"
            )

    # ------------------------------------------------------------
    # Return raw dataset if preprocessing disabled
    # ------------------------------------------------------------

    if not preprocess:

        logger.info("Returning raw German dataset.")

        return df

    # ------------------------------------------------------------
    # Prepare dataset
    # ------------------------------------------------------------

    logger.info("Running FairLens preprocessing...")

    processed_df, model, accuracy = prepare_dataset(
        df=df,
        target_column=target_column,
        sensitive_columns=sensitive_columns,
        prediction_column=prediction_column,
    )

    # Store metadata

    processed_df.attrs["model"] = model
    processed_df.attrs["baseline_accuracy"] = accuracy
    processed_df.attrs["dataset"] = "German Credit"

    logger.info(
        "German Credit preprocessing completed."
    )

    logger.info(
        "Baseline accuracy: %.4f",
        accuracy,
    )

    return processed_df

# ---------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------

def load_german_raw(path: str | Path) -> pd.DataFrame:
    """
    Load the raw German Credit dataset without preprocessing.

    Parameters
    ----------
    path : str or Path
        Path to the german.data file.

    Returns
    -------
    pandas.DataFrame
        Raw dataset.
    """
    path = Path(path)

    _validate_file(path)

    df = _load_raw_dataset(path)

    return _clean_dataset(df)


def dataset_info() -> dict:
    """
    Return metadata about the German Credit dataset.

    Returns
    -------
    dict
        Dataset information.
    """
    return {
        "name": "German Credit",
        "source": "UCI Statlog German Credit",
        "rows": 1000,
        "columns": len(GERMAN_COLUMNS),
        "target": TARGET_COLUMN,
        "prediction": PREDICTION_COLUMN,
        "default_sensitive_columns": list(DEFAULT_SENSITIVE_COLUMNS),
        "task": "Binary Classification",
        "license": "UCI Machine Learning Repository",
    }


# ---------------------------------------------------------------------
# Public Exports
# ---------------------------------------------------------------------

__all__ = [
    "GERMAN_COLUMNS",
    "TARGET_COLUMN",
    "PREDICTION_COLUMN",
    "DEFAULT_SENSITIVE_COLUMNS",
    "load_german_dataset",
    "load_german_raw",
    "dataset_info",
]