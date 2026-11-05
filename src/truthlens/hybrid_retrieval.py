def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60, top_n: int = 50) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, evidence_id in enumerate(ranking, start=1):
            scores[evidence_id] = scores.get(evidence_id, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores, key=lambda evidence_id: (-scores[evidence_id], evidence_id))
    return ordered[:top_n]


def fuse_rankings(
    bm25: dict[str, list[str]], dense: dict[str, list[str]], k: int = 60, top_n: int = 50
) -> dict[str, list[str]]:
    return {
        claim_id: reciprocal_rank_fusion([bm25[claim_id], dense[claim_id]], k, top_n)
        for claim_id in bm25
    }
