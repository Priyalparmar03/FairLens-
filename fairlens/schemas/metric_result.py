"""
Metric Result Schema

Defines the standard representation of a fairness metric
inside FairLens.

Author: FairLens
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass(slots=True)
class MetricResult:
    """
    Represents a single fairness metric.

    Parameters
    ----------
    name
        Canonical FairLens metric name.

    value
        Numerical metric value.

    threshold
        Fairness threshold.

    passed
        Whether the metric satisfies the threshold.

    library
        Library that produced this metric.

    description
        Human-readable explanation.
    """

    name: str
    value: float

    threshold: float

    passed: bool

    library: str

    description: str = ""

    unit: Optional[str] = None

    metadata: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------
    # Serialization
    # ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert MetricResult to dictionary.
        """

        return asdict(self)

    # ------------------------------------------------------
    # Pretty Representation
    # ------------------------------------------------------

    def __str__(self) -> str:

        status = "PASS" if self.passed else "FAIL"

        return (
            f"{self.name}: "
            f"{self.value:.4f} "
            f"({status})"
        )

    def __repr__(self) -> str:

        return (
            f"MetricResult("
            f"name='{self.name}', "
            f"value={self.value:.4f}, "
            f"passed={self.passed})"
        )