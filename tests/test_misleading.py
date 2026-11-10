import pandas as pd

from truthlens.misleading import build_misleading_set


def test_only_related_opposite_label_neighbors_are_used():
    claims = pd.DataFrame(
        {
            "claim_id": ["dev-0", "train-0", "train-1", "train-2"],
            "split": ["dev", "train", "train", "train"],
            "claim": [
                "texas taxes went up",
                "taxes in texas rose sharply",
                "taxes in texas rose",
                "moon landing was staged",
            ],
            "label_mapped": ["REFUTED", "SUPPORTED", "REFUTED", "SUPPORTED"],
        }
    )
    evidence = pd.DataFrame(
        {
            "claim_id": ["dev-0", "train-0", "train-1", "train-2"],
            "evidence_id": ["dev-0-0-0", "train-0-0-0", "train-1-0-0", "train-2-0-0"],
            "text": ["a", "b", "c", "d"],
        }
    )

    misleading = build_misleading_set(claims, evidence)

    assert misleading["neighbor_claim_id"].tolist() == ["train-0"]
    assert misleading["evidence_id"].tolist() == ["train-0-0-0"]
