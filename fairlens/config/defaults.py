"""
Default configuration for FairLens.

Author: FairLens
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(slots=True, frozen=True)
class FairLensConfig:
    """
    Global configuration for FairLens.
    """

    # --------------------------------------------------
    # Fairness
    # --------------------------------------------------

    fairness_threshold: float = 0.10

    disparate_impact_threshold: float = 0.80

    # --------------------------------------------------
    # Reports
    # --------------------------------------------------

    default_report_format: str = "markdown"

    supported_report_formats: List[str] = field(
        default_factory=lambda: [
            "markdown",
            "html",
            "json",
        ]
    )

    # --------------------------------------------------
    # Logging
    # --------------------------------------------------

    log_level: str = "INFO"

    # --------------------------------------------------
    # Metric Engines
    # --------------------------------------------------

    enable_fairlearn: bool = True

    enable_aif360: bool = True

    # --------------------------------------------------
    # MAI
    # --------------------------------------------------

    enable_metric_agreement_index: bool = True

    # --------------------------------------------------
    # EU AI Act
    # --------------------------------------------------

    enable_annex3_classifier: bool = True

    # --------------------------------------------------
    # Runtime
    # --------------------------------------------------

    random_seed: int = 42

    save_reports: bool = True

    report_directory: str = "reports"


DEFAULT_CONFIG = FairLensConfig()