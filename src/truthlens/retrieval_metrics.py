from statistics import mean


def recall_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & gold_ids) / len(gold_ids)


def precision_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    return len(set(ranked_ids[:k]) & gold_ids) / k


def reciprocal_rank(ranked_ids: list[str], gold_ids: set[str]) -> float:
    for rank, evidence_id in enumerate(ranked_ids, start=1):
        if evidence_id in gold_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(
    rankings: dict[str, list[str]],
    gold: dict[str, set[str]],
    ks: tuple[int, ...] = (1, 3, 5, 10, 20, 50),
) -> dict[str, float]:
    claim_ids = list(rankings)
    result = {"claims": len(claim_ids)}
    result["mrr"] = round(mean(reciprocal_rank(rankings[c], gold[c]) for c in claim_ids), 4)
    for k in ks:
        result[f"recall@{k}"] = round(
            mean(recall_at_k(rankings[c], gold[c], k) for c in claim_ids), 4
        )
        result[f"precision@{k}"] = round(
            mean(precision_at_k(rankings[c], gold[c], k) for c in claim_ids), 4
        )
    return result
