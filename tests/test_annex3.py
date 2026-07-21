from fairlens.annex3 import (
    Annex3Classifier,
)


def test_high_risk():

    classifier = Annex3Classifier()

    result = classifier.classify(
        "AI system used for recruitment."
    )

    assert result["high_risk"] is True


def test_non_high_risk():

    classifier = Annex3Classifier()

    result = classifier.classify(
        "Movie recommendation system."
    )

    assert result["high_risk"] is False