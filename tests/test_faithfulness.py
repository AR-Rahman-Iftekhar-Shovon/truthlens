import numpy as np

from truthlens.faithfulness import claim_faithfulness, drop_one, summarize_faithfulness


def test_drop_one():
    assert drop_one(["a", "b", "c"]) == [["b", "c"], ["a", "c"], ["a", "b"]]


def test_claim_faithfulness_finds_influential_passage():
    full = np.array([0.8, 0.1, 0.1])
    no_evidence = np.array([0.1, 0.8, 0.1])
    dropped = np.array([[0.75, 0.15, 0.1], [0.2, 0.7, 0.1], [0.78, 0.12, 0.1]])
    result = claim_faithfulness(full, no_evidence, dropped, [False, True, False])
    assert result["no_evidence_changes"] is True
    assert result["any_flip"] is True
    assert result["top_is_gold"] is True
    assert round(result["max_drop"], 2) == 0.6
    assert len(result["drops_gold"]) == 1 and len(result["drops_other"]) == 2


def test_summarize_faithfulness_chance_level():
    full = np.array([0.8, 0.1, 0.1])
    dropped = np.array([[0.7, 0.2, 0.1], [0.7, 0.2, 0.1]])
    row = claim_faithfulness(full, full, dropped, [True, False])
    summary = summarize_faithfulness([row])
    assert summary["chance_top_is_gold"] == 0.5
    assert summary["no_evidence_changes_verdict"] == 0.0
    assert summary["any_single_removal_flips_verdict"] == 0.0
