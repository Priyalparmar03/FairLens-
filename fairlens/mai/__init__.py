"""
Metric Agreement Index (MAI)

Novel contribution of FairLens.
"""

from .agreement import MetricAgreementIndex
from .weighting import MetricWeighting

__all__ = [
    "MetricAgreementIndex",
    "MetricWeighting",
]