from truthlens.rerank_data import build_training_pairs


def test_gold_are_positives_and_not_negatives():
    pairs, labels = build_training_pairs(
        {"c1": "claim"},
        {"e1": "gold", "e2": "bad", "e3": "worse"},
        {"c1": {"e1"}},
        {"c1": ["e1", "e2", "e3"]},
        negatives_per_claim=1,
    )

    assert pairs == [["claim", "gold"], ["claim", "bad"]]
    assert labels == [1, 0]
