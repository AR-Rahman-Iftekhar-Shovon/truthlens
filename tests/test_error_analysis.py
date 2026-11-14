from truthlens.error_analysis import error_category


def test_error_categories():
    assert error_category("REFUTED", "REFUTED", 0.9, 1) == "correct"
    assert error_category("REFUTED", "SUPPORTED", 0.9, 0) == "retrieval_failure"
    assert error_category("UNKNOWN", "REFUTED", 0.9, 2) == "insufficient_vs_verdict"
    assert error_category("REFUTED", "SUPPORTED", 0.8, 1) == "overconfident_flip"
    assert error_category("REFUTED", "SUPPORTED", 0.5, 1) == "low_confidence_flip"
