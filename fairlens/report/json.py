"""
JSON Report Generator

Produces structured JSON reports for FairLens.

These reports are intended for:
- APIs
- Dashboards
- CI/CD pipelines
- MLOps systems
- Governance platforms

Author: FairLens
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


class JSONReportGenerator:
    """
    Generate structured JSON fairness reports.
    """

    def generate(
        self,
        dataset_info: Dict[str, Any],
        fairlearn_metrics: Dict[str, Any],
        aif360_metrics: Dict[str, Any],
        mai: Dict[str, Any],
        annex3: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build a structured JSON report.

        Parameters
        ----------
        dataset_info
            Information about the dataset.

        fairlearn_metrics
            Metrics generated using Fairlearn.

        aif360_metrics
            Metrics generated using AIF360.

        mai
            Metric Agreement Index results.

        annex3
            Annex III classification.

        Returns
        -------
        dict
            Complete report.
        """

        report = {

            "metadata": {
                "generator": "FairLens",
                "version": "0.1.0",
                "generated_at": datetime.utcnow().isoformat() + "Z",
            },

            "dataset": dataset_info,

            "fairlearn": {
                "library": "Fairlearn",
                "metrics": fairlearn_metrics,
            },

            "aif360": {
                "library": "AIF360",
                "metrics": aif360_metrics,
            },

            "metric_agreement_index": mai,

            "eu_ai_act_annex3": annex3,
        }

        return report

    # =====================================================
    # Serialize
    # =====================================================

    def dumps(
        self,
        report: Dict[str, Any],
        indent: int = 4,
    ) -> str:
        """
        Convert report dictionary to JSON string.
        """

        return json.dumps(
            report,
            indent=indent,
            ensure_ascii=False,
        )

    # =====================================================
    # Save
    # =====================================================

    def save(
        self,
        path: str,
        report: Dict[str, Any],
        indent: int = 4,
    ) -> None:
        """
        Save report to disk.
        """

        Path(path).write_text(
            self.dumps(report, indent),
            encoding="utf-8",
        )

    # =====================================================
    # Load
    # =====================================================

    def load(
        self,
        path: str,
    ) -> Dict[str, Any]:
        """
        Load a previously generated report.
        """

        return json.loads(
            Path(path).read_text(
                encoding="utf-8"
            )
        )