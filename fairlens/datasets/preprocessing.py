
"""
fairlens.datasets.preprocessing

Shared preprocessing utilities for FairLens dataset loaders.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


def validate_dataset(
    df: pd.DataFrame,
    target_column: str,
    sensitive_columns: Optional[Sequence[str]] = None,
) -> None:
    """Validate required columns."""
    if df.empty:
        raise ValueError("Input dataframe is empty.")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    if sensitive_columns:
        missing = [c for c in sensitive_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing sensitive columns: {missing}")


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace common missing tokens with pandas NA."""
    df = df.copy()
    df.replace(["?", " ?", "", "NA", "N/A"], pd.NA, inplace=True)
    return df


def split_features_target(
    df: pd.DataFrame,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into X and y."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing pipeline."""
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [
        c for c in X.columns if c not in numeric_features
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def train_baseline_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Train a Logistic Regression baseline model.

    Returns
    -------
    model : sklearn.pipeline.Pipeline
    accuracy : float
    """
    preprocessor = build_preprocessor(X)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    logger.info("Baseline accuracy: %.4f", accuracy)

    return model, accuracy


def generate_predictions(
    model: Pipeline,
    X: pd.DataFrame,
    prediction_column: str = "prediction",
) -> pd.DataFrame:
    """Generate prediction column."""
    df = X.copy()
    df[prediction_column] = model.predict(X)
    return df


def prepare_dataset(
    df: pd.DataFrame,
    target_column: str,
    sensitive_columns: Optional[Sequence[str]] = None,
    prediction_column: str = "prediction",
):
    """
    Complete preprocessing workflow.

    Returns
    -------
    prepared_df : pd.DataFrame
    model : sklearn Pipeline
    accuracy : float
    """
    validate_dataset(df, target_column, sensitive_columns)

    df = clean_missing_values(df)

    X, y = split_features_target(df, target_column)

    model, accuracy = train_baseline_model(X, y)

    X_pred = generate_predictions(model, X, prediction_column)

    prepared_df = X_pred.copy()
    prepared_df[target_column] = y.values

    return prepared_df, model, accuracy


__all__: List[str] = [
    "validate_dataset",
    "clean_missing_values",
    "split_features_target",
    "build_preprocessor",
    "train_baseline_model",
    "generate_predictions",
    "prepare_dataset",
]
