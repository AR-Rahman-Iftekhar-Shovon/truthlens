from truthlens.report import abstention_reason, format_report, uncertainty_level


def test_abstention_reasons():
    assert abstention_reason("REFUTED", 0.9, 0.7, 0.75, 0.45) is None
    assert "weakly related" in abstention_reason("REFUTED", 0.9, 0.2, 0.75, 0.45)
    assert "insufficient" in abstention_reason("UNKNOWN", 0.9, 0.7, 0.75, 0.45)
    assert "below the threshold" in abstention_reason("SUPPORTED", 0.6, 0.7, 0.75, 0.45)


def test_uncertainty_levels():
    assert uncertainty_level(True, 0.99) == "High"
    assert uncertainty_level(False, 0.5) == "High"
    assert uncertainty_level(False, 0.7) == "Medium"
    assert uncertainty_level(False, 0.9) == "Low"


def test_report_contains_verdict_evidence_and_reason():
    evidence = [{"text": "Q: x A: y", "source_url": "http://a"}]
    probabilities = {"SUPPORTED": 0.1, "REFUTED": 0.8, "UNKNOWN": 0.1}

    verdict = format_report("claim", "REFUTED", 0.8, probabilities, evidence, 0.6, None)
    abstained = format_report(
        "claim", "INSUFFICIENT_EVIDENCE", 0.5, probabilities, evidence, 0.2, "weak evidence"
    )

    assert "Assessment: REFUTED" in verdict and "[http://a]" in verdict
    assert "Assessment: INSUFFICIENT_EVIDENCE" in abstained
    assert "Top guess (not a verdict): REFUTED 0.50" in abstained
    assert "Confidence: 0.80" in verdict
    assert "no verdict because weak evidence" in abstained
