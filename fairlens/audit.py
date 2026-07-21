"""Top-level orchestrator: runs a full audit and assembles the result."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from fairlens.datasets.loaders import Dataset
from fairlens.metrics.aif360_metrics import compute_aif360_metrics
from fairlens.metrics.fairlearn_metrics import compute_fairlearn_metrics
from fairlens.metrics.common import REGISTRY
from fairlens.mai.index import MetricAgreementIndex, MetricEvaluation, MAIResult
from fairlens.annex3.mapping import classify_annex3_risk, Annex3Classification


@dataclass
class AuditResult:
    dataset_name: str
    aif360_metrics: dict
    fairlearn_metrics: dict
    mai_result: MAIResult
    annex3: Optional[Annex3Classification] = None
    unfair_metric_keys: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset_name,
            "aif360_metrics": self.aif360_metrics,
            "fairlearn_metrics": self.fairlearn_metrics,
            "unfair_metrics": self.unfair_metric_keys,
            "mai": {
                "overall": self.mai_result.mai,
                "verdict_agreement": self.mai_result.mai_verdict,
                "rank_agreement": self.mai_result.mai_rank,
                "cross_library_agreement": self.mai_result.cross_library_mai,
                "cross_metric_agreement": self.mai_result.cross_metric_mai,
            },
            "annex3": (
                {
                    "risk_tier": self.annex3.risk_tier,
                    "matched_categories": self.annex3.matched_categories,
                    "relevant_articles": self.annex3.relevant_articles,
                    "rationale": self.annex3.rationale,
                    "fairness_flag": self.annex3.fairness_flag,
                }
                if self.annex3
                else None
            ),
        }


class FairnessAudit:
    """
    Runs model predictions through both AIF360 and Fairlearn, computes
    the canonical metric set from each, derives the Metric Agreement
    Index, and (optionally) maps the result onto EU AI Act Annex III.
    """

    def __init__(self, mai_weights: tuple[float, float] = (0.5, 0.5)):
        self.mai = MetricAgreementIndex(w_verdict=mai_weights[0], w_rank=mai_weights[1])

    def run(
        self,
        dataset: Dataset,
        y_pred: np.ndarray,
        use_case_description: Optional[str] = None,
    ) -> AuditResult:
        """
        `dataset`: a fairlens Dataset (see fairlens.datasets)
        `y_pred`: model predictions aligned with dataset.df rows
        `use_case_description`: optional free text; if given, the audit
            is also mapped to an EU AI Act Annex III risk tier.
        """
        y_pred = np.asarray(y_pred).reshape(-1)

        aif_metrics = compute_aif360_metrics(
            dataset.aif360_dataset,
            y_pred,
            privileged_groups=dataset.privileged_groups,
            unprivileged_groups=dataset.unprivileged_groups,
        )

        protected_attr = dataset.protected_attribute_names[0]
        sensitive = dataset.df[protected_attr].to_numpy()
        y_true = dataset.df[dataset.label_name].to_numpy()

        fl_metrics = compute_fairlearn_metrics(
            y_true=y_true, y_pred=y_pred, sensitive_features=sensitive
        )

        evaluations = []
        for key, value in aif_metrics.items():
            evaluations.append(MetricEvaluation(key, "aif360", value))
        for key, value in fl_metrics.items():
            evaluations.append(MetricEvaluation(key, "fairlearn", value))

        mai_result = self.mai.compute(evaluations)

        unfair_keys = sorted(
            {e.metric_key for e in mai_result.evaluations if not e.is_fair}
        )

        annex3 = None
        if use_case_description:
            annex3 = classify_annex3_risk(
                use_case_description,
                mai_result=mai_result,
                unfair_metric_keys=unfair_keys,
            )

        return AuditResult(
            dataset_name=dataset.name,
            aif360_metrics=aif_metrics,
            fairlearn_metrics=fl_metrics,
            mai_result=mai_result,
            annex3=annex3,
            unfair_metric_keys=unfair_keys,
        )
