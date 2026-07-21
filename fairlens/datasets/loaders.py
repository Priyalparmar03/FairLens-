"""
fairlens.datasets.loaders
=========================

Unified dataset loading interface for FairLens.

Supported datasets
------------------
- Adult Income
- COMPAS
- German Credit
- Synthetic (generated)
- Synthetic (CSV)

Example
-------
>>> from fairlens.datasets import load_dataset

>>> df = load_dataset(
...     name="adult",
...     path="data/adult/adult.data"
... )

>>> df = load_dataset(
...     name="compas",
...     path="data/compas/compas-scores-two-years.csv"
... )

>>> df = load_dataset(
...     name="german",
...     path="data/german/german.data"
... )

>>> df = load_dataset(
...     name="synthetic",
...     n_samples=5000
... )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .adult import load_adult_dataset
from .compas import load_compas_dataset
from .german import load_german_dataset
from .synthetic import (
    generate_synthetic_dataset,
    load_synthetic_dataset,
)

logger = logging.getLogger(__name__)


SUPPORTED_DATASETS = (
    "adult",
    "compas",
    "german",
    "synthetic",
)


def list_datasets() -> tuple[str, ...]:
    """
    Return all supported datasets.

    Returns
    -------
    tuple
    """
    return SUPPORTED_DATASETS


def load_dataset(
    name: str,
    *,
    path: str | Path | None = None,
    preprocess: bool = True,
    **kwargs: Any,
):
    """
    Generic dataset loader.

    Parameters
    ----------
    name : str
        Dataset name.

        Supported values

        - adult
        - compas
        - german
        - synthetic

    path : str or Path, optional
        Dataset path.

        Not required for generated synthetic datasets.

    preprocess : bool
        Whether preprocessing should be applied.

    kwargs
        Additional dataset-specific arguments.

    Returns
    -------
    pandas.DataFrame
    """

    dataset = name.lower().strip()

    logger.info("Loading dataset: %s", dataset)

    # ---------------------------------------------------------
    # Adult
    # ---------------------------------------------------------

    if dataset == "adult":

        if path is None:
            raise ValueError(
                "Adult dataset requires 'path'."
            )

        return load_adult_dataset(
            path=path,
            preprocess=preprocess,
            **kwargs,
        )

    # ---------------------------------------------------------
    # COMPAS
    # ---------------------------------------------------------

    if dataset == "compas":

        if path is None:
            raise ValueError(
                "COMPAS dataset requires 'path'."
            )

        return load_compas_dataset(
            path=path,
            preprocess=preprocess,
            **kwargs,
        )

    # ---------------------------------------------------------
    # German Credit
    # ---------------------------------------------------------

    if dataset == "german":

        if path is None:
            raise ValueError(
                "German dataset requires 'path'."
            )

        return load_german_dataset(
            path=path,
            preprocess=preprocess,
            **kwargs,
        )

    # ---------------------------------------------------------
    # Synthetic
    # ---------------------------------------------------------

    if dataset == "synthetic":

        if path is None:

            logger.info(
                "Generating synthetic dataset."
            )

            return generate_synthetic_dataset(
                preprocess=preprocess,
                **kwargs,
            )

        logger.info(
            "Loading synthetic dataset from CSV."
        )

        return load_synthetic_dataset(
            path=path,
            preprocess=preprocess,
            **kwargs,
        )

    # ---------------------------------------------------------
    # Unknown dataset
    # ---------------------------------------------------------

    raise ValueError(
        f"""
Unsupported dataset: {name}

Supported datasets:

{SUPPORTED_DATASETS}
"""
    )


__all__ = [
    "SUPPORTED_DATASETS",
    "list_datasets",
    "load_dataset",
]