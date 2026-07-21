"""
fairlens.datasets.synthetic

Synthetic dataset generation utilities for FairLens.

Supports
--------
• Generate synthetic fairness datasets
• Load synthetic datasets from CSV
• Automatic preprocessing
• Baseline model training
• Fairness benchmarking
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .preprocessing import prepare_dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------

TARGET_COLUMN = "Target"

PREDICTION_COLUMN = "prediction"

DEFAULT_SENSITIVE_COLUMNS = (
    "Gender",
    "Race",
    "Disability_Status",
)

DEFAULT_RANDOM_STATE = 42

# ---------------------------------------------------------------------
# Default categories
# ---------------------------------------------------------------------

GENDERS = [
    "Male",
    "Female",
]

RACES = [
    "White",
    "Black",
    "Asian",
    "Hispanic",
]

EDUCATION = [
    "High School",
    "Bachelor",
    "Master",
    "PhD",
]

OCCUPATIONS = [
    "Engineer",
    "Teacher",
    "Doctor",
    "Lawyer",
    "Business",
    "Student",
]

LOAN_PURPOSE = [
    "Education",
    "Business",
    "Home",
    "Vehicle",
]

REGIONS = [
    "North",
    "South",
    "East",
    "West",
]

MARITAL_STATUS = [
    "Single",
    "Married",
]

DISABILITY_STATUS = [
    "Yes",
    "No",
]

# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------

def _random_choice(
    rng: np.random.Generator,
    values,
    size,
):
    return rng.choice(values, size=size)



# ---------------------------------------------------------------------
# Synthetic Dataset Generator
# ---------------------------------------------------------------------

def generate_synthetic_dataset(
    n_samples: int = 10000,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    preprocess: bool = True,
    prediction_column: str = PREDICTION_COLUMN,
    target_column: str = TARGET_COLUMN,
    sensitive_columns: Optional[Sequence[str]] = DEFAULT_SENSITIVE_COLUMNS,
) -> pd.DataFrame:
    """
    Generate a realistic synthetic fairness dataset.

    Parameters
    ----------
    n_samples : int, default=10000
        Number of rows.

    random_state : int
        Random seed.

    preprocess : bool
        If True, generate prediction column using baseline model.

    Returns
    -------
    pandas.DataFrame
    """

    logger.info(
        "Generating synthetic dataset with %d samples...",
        n_samples,
    )

    rng = np.random.default_rng(random_state)

    age = rng.integers(18, 70, size=n_samples)

    gender = _random_choice(
        rng,
        GENDERS,
        n_samples,
    )

    race = _random_choice(
        rng,
        RACES,
        n_samples,
    )

    education = _random_choice(
        rng,
        EDUCATION,
        n_samples,
    )

    occupation = _random_choice(
        rng,
        OCCUPATIONS,
        n_samples,
    )

    annual_income = rng.normal(
        60000,
        18000,
        size=n_samples,
    ).astype(int)

    annual_income = np.clip(
        annual_income,
        15000,
        200000,
    )

    credit_score = rng.normal(
        680,
        80,
        size=n_samples,
    ).astype(int)

    credit_score = np.clip(
        credit_score,
        300,
        850,
    )

    employment_years = rng.integers(
        0,
        40,
        size=n_samples,
    )

    loan_amount = rng.normal(
        20000,
        9000,
        size=n_samples,
    ).astype(int)

    loan_amount = np.clip(
        loan_amount,
        1000,
        100000,
    )

    loan_purpose = _random_choice(
        rng,
        LOAN_PURPOSE,
        n_samples,
    )

    marital_status = _random_choice(
        rng,
        MARITAL_STATUS,
        n_samples,
    )

    dependents = rng.integers(
        0,
        6,
        size=n_samples,
    )

    region = _random_choice(
        rng,
        REGIONS,
        n_samples,
    )

    disability = _random_choice(
        rng,
        DISABILITY_STATUS,
        n_samples,
    )

    # ------------------------------------------------------------
    # Synthetic approval probability
    # ------------------------------------------------------------

    approval_score = (
        credit_score * 0.45
        + annual_income / 250
        + employment_years * 6
        - loan_amount / 350
    )

    probability = (
        approval_score - approval_score.min()
    ) / (
        approval_score.max() - approval_score.min()
    )

    target = np.where(
        probability > 0.50,
        "Approved",
        "Rejected",
    )

    df = pd.DataFrame(
        {
            "Age": age,
            "Gender": gender,
            "Race": race,
            "Education": education,
            "Occupation": occupation,
            "Annual_Income": annual_income,
            "Credit_Score": credit_score,
            "Employment_Years": employment_years,
            "Loan_Amount": loan_amount,
            "Loan_Purpose": loan_purpose,
            "Marital_Status": marital_status,
            "Dependents": dependents,
            "Region": region,
            "Disability_Status": disability,
            target_column: target,
        }
    )

    logger.info(
        "Synthetic dataset created successfully."
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
    processed_df.attrs["dataset"] = "Synthetic"

    logger.info(
        "Synthetic preprocessing completed."
    )

    logger.info(
        "Baseline accuracy: %.4f",
        accuracy,
    )

    return processed_df


# ---------------------------------------------------------------------
# Load Existing Synthetic Dataset
# ---------------------------------------------------------------------

def load_synthetic_dataset(
    path: str | Path,
    *,
    preprocess: bool = True,
    target_column: str = TARGET_COLUMN,
    prediction_column: str = PREDICTION_COLUMN,
    sensitive_columns: Optional[Sequence[str]] = DEFAULT_SENSITIVE_COLUMNS,
) -> pd.DataFrame:
    """
    Load an existing synthetic dataset from CSV.

    Parameters
    ----------
    path : str or Path
        Path to synthetic CSV file.

    preprocess : bool, default=True
        Whether to preprocess the dataset and generate predictions.

    Returns
    -------
    pandas.DataFrame
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Synthetic dataset not found: {path}"
        )

    logger.info("Loading synthetic dataset from %s", path)

    df = pd.read_csv(path)

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
    processed_df.attrs["dataset"] = "Synthetic"

    return processed_df


# ---------------------------------------------------------------------
# Save Dataset
# ---------------------------------------------------------------------

def save_synthetic_dataset(
    df: pd.DataFrame,
    path: str | Path,
) -> None:
    """
    Save a synthetic dataset to CSV.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to save.

    path : str or Path
        Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)

    logger.info("Synthetic dataset saved to %s", path)


# ---------------------------------------------------------------------
# Dataset Metadata
# ---------------------------------------------------------------------

def dataset_info() -> dict:
    """
    Return metadata describing the synthetic dataset.
    """
    return {
        "name": "Synthetic Fairness Dataset",
        "generator": "FairLens",
        "target": TARGET_COLUMN,
        "prediction": PREDICTION_COLUMN,
        "default_sensitive_columns": list(DEFAULT_SENSITIVE_COLUMNS),
        "supported_sensitive_columns": [
            "Gender",
            "Race",
            "Disability_Status",
            "Age",
        ],
        "task": "Binary Classification",
    }


# ---------------------------------------------------------------------
# Public Exports
# ---------------------------------------------------------------------

__all__ = [
    "TARGET_COLUMN",
    "PREDICTION_COLUMN",
    "DEFAULT_SENSITIVE_COLUMNS",
    "generate_synthetic_dataset",
    "load_synthetic_dataset",
    "save_synthetic_dataset",
    "dataset_info",
]