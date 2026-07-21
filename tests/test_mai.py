from fairlens.mai import (
    MetricAgreementIndex,
)


def test_metric_agreement():

    fairlearn = {
        "dp": 0.11,
        "eo": 0.08,
    }

    aif360 = {
        "dp": 0.12,
        "eo": 0.09,
    }

    mai = MetricAgreementIndex()

    result = mai.compute(
        fairlearn,
        aif360,
    )

    assert (
        0
        <= result["overall_agreement"]
        <= 1
    )

    assert "metric_agreements" in result