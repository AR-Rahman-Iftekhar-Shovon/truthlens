from truthlens.retrieval_metrics import (
    evaluate_retrieval,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_precision_and_reciprocal_rank():
    ranked = ["a", "b", "c", "d"]
    gold = {"b", "d"}

    assert recall_at_k(ranked, gold, 2) == 0.5
    assert precision_at_k(ranked, gold, 2) == 0.5
    assert reciprocal_rank(ranked, gold) == 0.5


def test_evaluate_retrieval_averages_over_claims():
    rankings = {"c1": ["a", "b"], "c2": ["x", "y"]}
    gold = {"c1": {"a"}, "c2": {"y"}}

    result = evaluate_retrieval(rankings, gold, ks=(1, 2))

    assert result["mrr"] == 0.75
    assert result["recall@1"] == 0.5
    assert result["recall@2"] == 1.0
