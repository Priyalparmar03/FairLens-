from fairlens.metrics import (
    FairlearnMetricEngine,
)


def test_fairlearn_metrics(
    sample_dataframe,
):

    engine = FairlearnMetricEngine()

    results = engine.compute(
        sample_dataframe,
        target_column="target",
        prediction_column="prediction",
        sensitive_feature="sex",
    )

    assert len(results.metrics) > 0

    for metric in results.metrics:

        assert metric.name != ""

        assert isinstance(
            metric.value,
            float,
        )