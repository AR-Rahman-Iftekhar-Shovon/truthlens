from truthlens.evaluation import summarize


def test_summarize_perfect_predictions():
    summary = summarize(["a", "b", "a"], ["a", "b", "a"], labels=["a", "b"])

    assert summary["accuracy"] == 1.0
    assert summary["macro_f1"] == 1.0
    assert summary["confusion_matrix"] == [[2, 0], [0, 1]]
