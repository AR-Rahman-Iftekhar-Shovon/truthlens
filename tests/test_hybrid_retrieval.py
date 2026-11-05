from truthlens.hybrid_retrieval import reciprocal_rank_fusion


def test_rrf_is_deterministic_on_ties():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60, top_n=3)

    assert fused == ["a", "b", "c"]


def test_rrf_prefers_passage_found_by_both_rankings():
    fused = reciprocal_rank_fusion([["x", "a"], ["a", "y"]], k=60, top_n=3)

    assert fused[0] == "a"
