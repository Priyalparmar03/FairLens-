"""
FairLens: an open-source framework for auditing ML models for fairness.

Benchmarks AIF360 vs Fairlearn metric implementations side by side, maps
audit results to EU AI Act Annex III risk tiers, and introduces the
Metric Agreement Index (MAI) -- a novel measure of how much different
fairness metrics and toolkits agree on whether a model is "fair".
"""

from fairlens.audit import FairnessAudit, AuditResult
from fairlens.mai.index import MetricAgreementIndex
from fairlens.annex3.mapping import classify_annex3_risk

__version__ = "0.1.0"

__all__ = [
    "FairnessAudit",
    "AuditResult",
    "MetricAgreementIndex",
    "classify_annex3_risk",
]
