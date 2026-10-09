import pytest

from truthlens.baseline import build_tfidf_classifier, build_tfidf_logreg

TEXTS = ["taxes rise fast", "taxes rise slow", "jobs grow fast", "jobs grow slow"] * 2
LABELS = ["false", "false", "true", "true"] * 2


def test_logreg_baseline_fits_and_predicts_known_labels():
    model = build_tfidf_logreg(seed=42)
    model.fit(TEXTS, LABELS)

    assert list(model.predict(["taxes rise fast", "jobs grow slow"])) == ["false", "true"]


def test_linear_svm_fits_and_predicts_known_labels():
    model = build_tfidf_classifier("linear_svm", 1.0, seed=42)
    model.fit(TEXTS, LABELS)

    assert list(model.predict(["taxes rise fast", "jobs grow slow"])) == ["false", "true"]


def test_unknown_classifier_kind_raises():
    with pytest.raises(ValueError):
        build_tfidf_classifier("random_forest", 1.0, seed=42)
