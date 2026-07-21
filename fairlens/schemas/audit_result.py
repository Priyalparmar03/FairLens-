"""
Audit Result Schema

Represents the complete output of a FairLens audit.

Author: FairLens
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List

from .metric_result import MetricResult


# ==========================================================
# Audit Metadata
# ==========================================================

@dataclass(slots=True)
class AuditMetadata:
    """
    Metadata about the audit execution.
    """

    dataset_name: str

    protected_attribute: str

    target_column: str

    prediction_column: str

    generated_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )

    fairlens_version: str = "0.1.0"

    python_version: str = ""

    execution_time_ms: float = 0.0


# ==========================================================
# Audit Result
# ==========================================================

@dataclass(slots=True)
class AuditResult:
    """
    Complete FairLens audit result.
    """

    metadata: AuditMetadata

    fairlearn_metrics: List[MetricResult] = field(
        default_factory=list
    )

    aif360_metrics: List[MetricResult] = field(
        default_factory=list
    )

    mai: Dict[str, Any] = field(
        default_factory=dict
    )

    annex3: Dict[str, Any] = field(
        default_factory=dict
    )

    summary: Dict[str, Any] = field(
        default_factory=dict
    )

    recommendations: List[str] = field(
        default_factory=list
    )

    # ------------------------------------------------------
    # Add Metrics
    # ------------------------------------------------------

    def add_fairlearn_metric(
        self,
        metric: MetricResult,
    ) -> None:

        self.fairlearn_metrics.append(metric)

    def add_aif360_metric(
        self,
        metric: MetricResult,
    ) -> None:

        self.aif360_metrics.append(metric)

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    @property
    def total_metrics(self) -> int:

        return (
            len(self.fairlearn_metrics)
            + len(self.aif360_metrics)
        )

    @property
    def failed_metrics(self) -> int:

        total = 0

        for metric in self.fairlearn_metrics:

            if not metric.passed:
                total += 1

        for metric in self.aif360_metrics:

            if not metric.passed:
                total += 1

        return total

    @property
    def passed_metrics(self) -> int:

        return self.total_metrics - self.failed_metrics

    # ------------------------------------------------------
    # Serialize
    # ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:

        return {

            "metadata": asdict(
                self.metadata
            ),

            "fairlearn_metrics": [
                m.to_dict()
                for m in self.fairlearn_metrics
            ],

            "aif360_metrics": [
                m.to_dict()
                for m in self.aif360_metrics
            ],

            "mai": self.mai,

            "annex3": self.annex3,

            "summary": self.summary,

            "recommendations": self.recommendations,
        }

    # ------------------------------------------------------
    # Pretty Print
    # ------------------------------------------------------

    def __str__(self):

        return (
            f"AuditResult("
            f"metrics={self.total_metrics}, "
            f"failed={self.failed_metrics}, "
            f"mai={self.mai.get('overall_agreement', 'N/A')})"
        )