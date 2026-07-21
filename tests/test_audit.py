import numpy as np
import pytest

from fairlens.datasets import load_synthetic
from fairlens.audit import FairnessAudit
from fairlens.mai.index import MetricAgreementIndex, MetricEvaluation
from fairlens.annex3.mapping import classify_annex3_risk
from fairlens.metrics.common import REGISTRY


def _predict_from_true_label_with_flip(dataset, flip_rate=0.0, seed=0):
    """Cheap stand-in 'model': true label with some noise, so metrics
    have something non-trivial to compute over."""
    rng = np.random.RandomState(seed)
    y = dataset.df[dataset.label_name].to_numpy().copy()
    flip_mask = rng.rand(len(y)) < flip_rate
    y[flip_mask] = 1 - y[flip_mask]
    return y


class TestSyntheticDatasetGeneration:
    def test_low_bias_dataset_shapes(self):
        ds = load_synthetic(n_samples=500, bias_strength=0.0, random_state=1)
        assert len(ds.df) == 500
        assert set(ds.df["protected"].unique()) <= {0, 1}

    def test_bias_strength_changes_label_distribution(self):
        low = load_synthetic(n_samples=3000, bias_strength=0.0, random_state=1)
        high = load_synthetic(n_samples=3000, bias_strength=1.0, random_state=1)

        def rate_gap(ds):
            g = ds.df.groupby("protected")[ds.label_name].mean()
            return abs(g.loc[1] - g.loc[0])

        assert rate_gap(high) > rate_gap(low)


class TestFairnessAudit:
    def test_full_audit_runs_on_synthetic_low_bias(self):
        ds = load_synthetic(n_samples=2000, bias_strength=0.0, random_state=2)
        y_pred = _predict_from_true_label_with_flip(ds, flip_rate=0.05)

        audit = FairnessAudit()
        result = audit.run(ds, y_pred, use_case_description="loan approval credit scoring")

        assert result.aif360_metrics
        assert result.fairlearn_metrics
        assert 0.0 <= result.mai_result.mai <= 1.0
        assert result.annex3 is not None
        assert result.annex3.risk_tier == "high-risk (Annex III)"

    def test_high_bias_dataset_flags_more_unfair_metrics_than_low_bias(self):
        ds_low = load_synthetic(n_samples=3000, bias_strength=0.0, random_state=3)
        ds_high = load_synthetic(n_samples=3000, bias_strength=1.0, random_state=3)

        y_low = _predict_from_true_label_with_flip(ds_low, flip_rate=0.02)
        y_high = _predict_from_true_label_with_flip(ds_high, flip_rate=0.02)

        audit = FairnessAudit()
        r_low = audit.run(ds_low, y_low)
        r_high = audit.run(ds_high, y_high)

        assert len(r_high.unfair_metric_keys) >= len(r_low.unfair_metric_keys)

    def test_result_serializes_to_dict(self):
        ds = load_synthetic(n_samples=500, bias_strength=0.2, random_state=4)
        y_pred = _predict_from_true_label_with_flip(ds, flip_rate=0.05)
        result = FairnessAudit().run(ds, y_pred)
        d = result.to_dict()
        assert "mai" in d and "aif360_metrics" in d and "fairlearn_metrics" in d


class TestMetricAgreementIndex:
    def test_perfect_agreement_gives_mai_near_one(self):
        evals = [
            MetricEvaluation("statistical_parity_difference", "aif360", 0.0),
            MetricEvaluation("statistical_parity_difference", "fairlearn", 0.0),
            MetricEvaluation("disparate_impact", "aif360", 1.0),
            MetricEvaluation("disparate_impact", "fairlearn", 1.0),
        ]
        result = MetricAgreementIndex().compute(evals)
        assert result.mai == pytest.approx(1.0)
        assert result.cross_library_mai == pytest.approx(1.0)

    def test_maximal_disagreement_gives_low_mai(self):
        evals = [
            MetricEvaluation("statistical_parity_difference", "aif360", 0.0),   # fair
            MetricEvaluation("statistical_parity_difference", "fairlearn", 0.8),  # very unfair
            MetricEvaluation("disparate_impact", "aif360", 1.0),                # fair
            MetricEvaluation("disparate_impact", "fairlearn", 0.1),             # very unfair
        ]
        result = MetricAgreementIndex().compute(evals)
        assert result.mai < 0.5

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError):
            MetricAgreementIndex(w_verdict=0.6, w_rank=0.6)

    def test_rank_agreement_none_below_three_evaluations(self):
        evals = [
            MetricEvaluation("statistical_parity_difference", "aif360", 0.0),
            MetricEvaluation("disparate_impact", "aif360", 1.0),
        ]
        result = MetricAgreementIndex().compute(evals)
        assert result.mai_rank is None


class TestAnnex3Mapping:
    def test_matches_employment_category(self):
        c = classify_annex3_risk("resume screening tool for job applicants")
        assert c.risk_tier == "high-risk (Annex III)"
        assert any(name == "Employment, workers management, self-employment"
                    for _, name in c.matched_categories)

    def test_unmatched_use_case(self):
        c = classify_annex3_risk("internal recommendation engine for stock photos")
        assert c.risk_tier == "unclassified -- review manually"

    def test_fairness_flag_present_when_unfair_metrics_given(self):
        c = classify_annex3_risk(
            "loan approval credit scoring",
            unfair_metric_keys=["disparate_impact"],
        )
        assert c.fairness_flag is not None


class TestMetricRegistry:
    def test_is_fair_difference_metric(self):
        spec = REGISTRY["statistical_parity_difference"]
        assert spec.is_fair(0.05)
        assert not spec.is_fair(0.5)

    def test_is_fair_ratio_metric(self):
        spec = REGISTRY["disparate_impact"]
        assert spec.is_fair(0.9)
        assert not spec.is_fair(0.5)
