import pandas as pd

from truthlens.verification_data import build_examples


def test_conflicting_excluded_and_evidence_truncated():
    claims = pd.DataFrame(
        {
            "claim_id": ["c1", "c2"],
            "claim": ["a", "b"],
            "label_mapped": ["SUPPORTED", "CONFLICTING"],
        }
    )

    examples = build_examples(claims, {"c1": ["p1", "p2", "p3"], "c2": ["p4"]}, max_passages=2)

    assert examples["claim_id"].tolist() == ["c1"]
    assert examples["evidence"].tolist() == ["p1 p2"]
