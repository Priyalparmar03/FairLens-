"""
FairLens Command Line Interface.

Example

fairlens audit \
    --data adult.csv \
    --target income \
    --prediction prediction \
    --protected sex \
    --report html

Author: FairLens
"""

from __future__ import annotations

import argparse

import pandas as pd

from fairlens.audit import FairnessAuditor
from fairlens.report.markdown import MarkdownReportGenerator
from fairlens.report.html import HTMLReportGenerator
from fairlens.report.json import JSONReportGenerator


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        prog="fairlens",
        description="FairLens Fairness Audit Toolkit",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    audit = subparsers.add_parser(
        "audit",
        help="Run fairness audit",
    )

    audit.add_argument(
        "--data",
        required=True,
        help="CSV dataset",
    )

    audit.add_argument(
        "--target",
        required=True,
    )

    audit.add_argument(
        "--prediction",
        required=True,
    )

    audit.add_argument(
        "--protected",
        required=True,
    )

    audit.add_argument(
        "--report",
        default="markdown",
        choices=[
            "markdown",
            "html",
            "json",
        ],
    )

    audit.add_argument(
        "--output",
        default="fairness_report",
    )

    return parser


def run_audit(args):

    df = pd.read_csv(args.data)

    auditor = FairnessAuditor()

    result = auditor.audit(
        dataframe=df,
        target_column=args.target,
        prediction_column=args.prediction,
        sensitive_feature=args.protected,
    )

    if args.report == "markdown":

        generator = MarkdownReportGenerator()

        report = generator.generate(
            result.metadata,
            result.fairlearn_metrics,
            result.aif360_metrics,
            result.mai,
            result.annex3,
        )

        generator.save(
            args.output + ".md",
            report,
        )

    elif args.report == "html":

        generator = HTMLReportGenerator()

        report = generator.generate(
            result.metadata,
            result.fairlearn_metrics,
            result.aif360_metrics,
            result.mai,
            result.annex3,
        )

        generator.save(
            args.output + ".html",
            report,
        )

    else:

        generator = JSONReportGenerator()

        report = generator.generate(
            result.metadata,
            result.fairlearn_metrics,
            result.aif360_metrics,
            result.mai,
            result.annex3,
        )

        generator.save(
            args.output + ".json",
            report,
        )

    print()

    print("=" * 50)

    print(" FairLens Audit Completed")

    print("=" * 50)

    print()

    print("Report saved as:", args.output)


def main():

    parser = build_parser()

    args = parser.parse_args()

    if args.command == "audit":

        run_audit(args)


if __name__ == "__main__":

    main()