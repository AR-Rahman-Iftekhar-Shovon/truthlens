from truthlens.live_eval import score_live


def row(label, verdict):
    return {"label": label, "verdict": verdict}


def test_score_live_counts():
    rows = [
        row("SUPPORTED", "SUPPORTED"),
        row("REFUTED", "SUPPORTED"),
        row("REFUTED", "INSUFFICIENT_EVIDENCE"),
        row("INSUFFICIENT_EVIDENCE", "INSUFFICIENT_EVIDENCE"),
    ]
    result = score_live(rows)
    assert result["accuracy"] == 0.5
    assert result["coverage"] == 0.5
    assert result["accuracy_on_answered"] == 0.5
    assert result["wrong_answers"] == 1
    assert result["by_label"]["REFUTED"]["correct"] == 0


def test_score_live_all_abstain():
    result = score_live([row("SUPPORTED", "INSUFFICIENT_EVIDENCE")])
    assert result["accuracy_on_answered"] is None
    assert result["coverage"] == 0.0
