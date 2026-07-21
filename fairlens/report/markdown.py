"""
Markdown Report Generator

Generates a human-readable fairness audit report.

Author: FairLens
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List


class MarkdownReportGenerator:
    """
    Generate Markdown fairness reports.

    Sections
    --------
    - Summary
    - Dataset Information
    - Fairlearn Metrics
    - AIF360 Metrics
    - Metric Agreement Index
    - Annex III Classification
    - Recommendations
    """

    def __init__(self):

        self.lines: List[str] = []

    # ======================================================
    # Helpers
    # ======================================================

    def _add(self, text: str = "") -> None:

        self.lines.append(text)

    def _header(self):

        self._add("# FairLens Fairness Audit Report")
        self._add("")
        self._add(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self._add("")
        self._add("---")
        self._add("")

    # ======================================================
    # Dataset
    # ======================================================

    def _dataset_section(
        self,
        dataset_info: Dict,
    ):

        self._add("## Dataset Information")
        self._add("")

        for key, value in dataset_info.items():

            self._add(f"- **{key}**: {value}")

        self._add("")

    # ======================================================
    # Metric Table
    # ======================================================

    def _metric_table(
        self,
        title: str,
        metrics: Dict,
    ):

        self._add(f"## {title}")
        self._add("")

        self._add(
            "| Metric | Value | Threshold | Pass |"
        )

        self._add(
            "|-------|-------|-----------|------|"
        )

        for metric in metrics.values():

            passed = "✅" if metric["passed"] else "❌"

            self._add(
                f"| {metric['name']} | "
                f"{metric['value']:.4f} | "
                f"{metric['threshold']} | "
                f"{passed} |"
            )

        self._add("")

    # ======================================================
    # MAI
    # ======================================================

    def _mai_section(
        self,
        mai: Dict,
    ):

        self._add("## Metric Agreement Index")
        self._add("")

        self._add(
            f"Overall Agreement: **{mai['overall_agreement']:.4f}**"
        )

        self._add("")

        self._add("| Metric | Agreement |")
        self._add("|-------|-----------|")

        for metric, score in mai[
            "metric_agreements"
        ].items():

            self._add(
                f"| {metric} | {score:.4f} |"
            )

        self._add("")

    # ======================================================
    # Annex III
    # ======================================================

    def _annex_section(
        self,
        annex: Dict,
    ):

        self._add("## EU AI Act Annex III")
        self._add("")

        self._add(
            f"High Risk: **{annex['high_risk']}**"
        )

        self._add("")

        if annex["categories"]:

            for category in annex["categories"]:

                self._add(
                    f"- {category['name']}"
                )

        self._add("")
        self._add(
            f"Recommendation: {annex['recommendation']}"
        )

        self._add("")

    # ======================================================
    # Recommendations
    # ======================================================

    def _recommendations(
        self,
        fairlearn: Dict,
        aif360: Dict,
    ):

        self._add("## Recommendations")
        self._add("")

        recommendations = []

        for metric in fairlearn.values():

            if not metric["passed"]:

                recommendations.append(
                    f"Improve **{metric['name']}** "
                    "using bias mitigation techniques."
                )

        if not recommendations:

            recommendations.append(
                "No major fairness concerns detected."
            )

        for rec in recommendations:

            self._add(f"- {rec}")

        self._add("")

    # ======================================================
    # Main API
    # ======================================================

    def generate(
        self,
        dataset_info: Dict,
        fairlearn_metrics: Dict,
        aif360_metrics: Dict,
        mai: Dict,
        annex3: Dict,
    ) -> str:

        self.lines.clear()

        self._header()

        self._dataset_section(dataset_info)

        self._metric_table(
            "Fairlearn Metrics",
            fairlearn_metrics,
        )

        self._metric_table(
            "AIF360 Metrics",
            aif360_metrics,
        )

        self._mai_section(mai)

        self._annex_section(annex3)

        self._recommendations(
            fairlearn_metrics,
            aif360_metrics,
        )

        return "\n".join(self.lines)

    # ======================================================
    # Save
    # ======================================================

    def save(
        self,
        path: str,
        report: str,
    ):

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as f:

            f.write(report)