import pytest

from truthlens.baseline import build_tfidf_classifier, build_tfidf_logreg
from truthlens.features import top_weighted_features

TEXTS = ["taxes rise fast", "taxes rise slow", "jobs grow fast", "jobs grow slow"] * 2
LABELS = ["false", "false", "true", "true"] * 2


def test_top_weighted_features_point_to_expected_classes():
    model = build_tfidf_logreg(seed=42)
    model.fit(TEXTS, LABELS)

    highest, lowest = top_weighted_features(model, 2)

    assert set(highest) <= {"jobs", "grow", "jobs grow", "grow fast", "grow slow"}
    assert set(lowest) <= {"taxes", "rise", "taxes rise", "rise fast", "rise slow"}


def test_multiclass_model_is_rejected():
    model = build_tfidf_classifier("logreg", 1.0, seed=42)
    model.fit(TEXTS, ["a", "b", "c", "a"] * 2)

    with pytest.raises(ValueError):
        top_weighted_features(model, 2)