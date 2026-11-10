import numpy as np


def rerank(
    claim_texts: dict[str, str],
    candidates: dict[str, list[str]],
    passages: dict[str, str],
    model,
    top_n: int = 50,
    batch_size: int = 64,
) -> dict[str, list[str]]:
    reranked = {}
    for claim_id, evidence_ids in candidates.items():
        pairs = [[claim_texts[claim_id], passages[evidence_id]] for evidence_id in evidence_ids]
        scores = np.asarray(model.predict(pairs, batch_size=batch_size, show_progress_bar=False))
        order = np.argsort(-scores, kind="stable")[:top_n]
        reranked[claim_id] = [evidence_ids[i] for i in order]
    return reranked
