def build_training_pairs(
    claim_texts: dict[str, str],
    passages: dict[str, str],
    gold: dict[str, set[str]],
    rankings: dict[str, list[str]],
    negatives_per_claim: int = 8,
) -> tuple[list[list[str]], list[int]]:
    pairs, labels = [], []
    for claim_id, ranked_ids in rankings.items():
        for evidence_id in sorted(gold[claim_id]):
            pairs.append([claim_texts[claim_id], passages[evidence_id]])
            labels.append(1)
        negatives = [e for e in ranked_ids if e not in gold[claim_id]][:negatives_per_claim]
        for evidence_id in negatives:
            pairs.append([claim_texts[claim_id], passages[evidence_id]])
            labels.append(0)
    return pairs, labels
