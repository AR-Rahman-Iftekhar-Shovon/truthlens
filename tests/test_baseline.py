from truthlens.baseline import build_tfidf_logreg


def test_baseline_fits_and_predicts_known_labels():
    texts = ["taxes rise fast", "taxes rise slow", "jobs grow fast", "jobs grow slow"] * 2
    labels = ["false", "false", "true", "true"] * 2

    model = build_tfidf_logreg(seed=42)
    model.fit(texts, labels)
    predictions = model.predict(["taxes rise fast", "jobs grow slow"])

    assert list(predictions) == ["false", "true"]
