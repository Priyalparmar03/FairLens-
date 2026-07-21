from fairlens.report.json import (
    JSONReportGenerator,
)


def test_json_report(dataset_info):

    generator = JSONReportGenerator()

    report = generator.generate(
        dataset_info,
        {},
        {},
        {},
        {},
    )

    assert "metadata" in report

    assert "dataset" in report

    assert "fairlearn" in report


def test_json_serialization(dataset_info):

    generator = JSONReportGenerator()

    report = generator.generate(
        dataset_info,
        {},
        {},
        {},
        {},
    )

    text = generator.dumps(report)

    assert isinstance(
        text,
        str,
    )