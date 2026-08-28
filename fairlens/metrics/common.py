from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MetricSpec:
    key: str                       
    display_name: str
    kind: Literal["difference", "ratio"]
    ideal: float
    default_threshold: float       
    aif360_name: str | None
    fairlearn_name: str | None
    description: str

    def is_fair(self, value: float, threshold: float | None = None) -> bool:
        t = self.default_threshold if threshold is None else threshold
        if self.kind == "difference":
            return abs(value - self.ideal) <= t
        lower = t
        upper = 1.0 / t if t > 0 else float("inf")
        return lower <= value <= upper


REGISTRY: dict[str, MetricSpec] = {
    "statistical_parity_difference": MetricSpec(
        key="statistical_parity_difference",
        display_name="Statistical Parity Difference",
        kind="difference",
        ideal=0.0,
        default_threshold=0.10,
        aif360_name="statistical_parity_difference",
        fairlearn_name="demographic_parity_difference",
        description=(
            "P(favorable | unprivileged) - P(favorable | privileged). "
            "Measures demographic (independence) parity. NOTE: AIF360's "
            "version is signed (can be negative); Fairlearn's "
            "demographic_parity_difference is unsigned (max-min group "
            "selection rate, always >= 0). FairLens reports both "
            "as-computed -- a sign mismatch between the two here is an "
            "expected convention difference, not an error."
        ),
    ),
    "disparate_impact": MetricSpec(
        key="disparate_impact",
        display_name="Disparate Impact",
        kind="ratio",
        ideal=1.0,
        default_threshold=0.80,  # the "80% rule"
        aif360_name="disparate_impact",
        fairlearn_name="demographic_parity_ratio",
        description=(
            "P(favorable | unprivileged) / P(favorable | privileged). "
            "The legal '80% rule' threshold is the common convention."
        ),
    ),
    "equal_opportunity_difference": MetricSpec(
        key="equal_opportunity_difference",
        display_name="Equal Opportunity Difference",
        kind="difference",
        ideal=0.0,
        default_threshold=0.10,
        aif360_name="equal_opportunity_difference",
        fairlearn_name="equal_opportunity_difference",  # computed manually, see fairlearn_metrics.py
        description=(
            "TPR(unprivileged) - TPR(privileged). Measures separation "
            "restricted to the true-positive class (Hardt et al. 2016)."
        ),
    ),
    "average_odds_difference": MetricSpec(
        key="average_odds_difference",
        display_name="Average Odds Difference",
        kind="difference",
        ideal=0.0,
        default_threshold=0.10,
        aif360_name="average_odds_difference",
        fairlearn_name="equalized_odds_difference",  # closest Fairlearn analogue
        description=(
            "Average of the TPR and FPR gaps between unprivileged and "
            "privileged groups. AIF360's signed version vs Fairlearn's "
            "unsigned equalized_odds_difference (max of the two gaps) -- "
            "these are related but not numerically identical; FairLens "
            "flags this as an implementation-choice divergence, not a bug."
        ),
    ),
    "theil_index": MetricSpec(
        key="theil_index",
        display_name="Theil Index",
        kind="difference",
        ideal=0.0,
        default_threshold=0.10,
        aif360_name="theil_index",
        fairlearn_name=None,  
        description=(
            "Generalized entropy index of individual (not group) fairness "
            "in the benefit distribution. AIF360-only; included for "
            "within-library cross-metric agreement analysis."
        ),
    ),
}


def cross_library_metrics() -> list[str]:
    """Canonical keys computable by both AIF360 and Fairlearn."""
    return [k for k, v in REGISTRY.items() if v.aif360_name and v.fairlearn_name]
