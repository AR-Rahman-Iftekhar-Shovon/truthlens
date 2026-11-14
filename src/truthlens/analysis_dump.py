from truthlens.bm25_retrieval import retrieve as bm25_retrieve
from truthlens.dense_retrieval import embed, retrieve_with_scores
from truthlens.hybrid_retrieval import fuse_rankings
from truthlens.rerank import rerank
from truthlens.robustness import evidence_conditions
from truthlens.verification_data import build_examples

DEPTH = 100
TOP_K = 5


def max_similarity(
    model, claim_texts: dict[str, str], evidence_texts: dict[str, list[str]]
) -> dict:
    scores = {}
    for claim_id, passages in evidence_texts.items():
        if not passages:
            scores[claim_id] = 0.0
            continue
        vectors = embed(model, [claim_texts[claim_id]] + passages)
        scores[claim_id] = round(float((vectors[1:] @ vectors[0]).max()), 4)
    return scores


def collect(dev_claims, evidence, passages, dense_model, cross_encoder, misleading, seed, verify):
    claim_texts = dict(zip(dev_claims["claim_id"], dev_claims["claim"]))
    dense_scored = retrieve_with_scores(dev_claims, evidence, dense_model, k=DEPTH)
    dense = {claim_id: [e for e, _ in ranked] for claim_id, ranked in dense_scored.items()}
    bm25 = bm25_retrieve(dev_claims, evidence, k=DEPTH)
    hybrid = fuse_rankings(bm25, dense, top_n=50)
    dense_top50 = {claim_id: ids[:50] for claim_id, ids in dense.items()}
    reranked = rerank(claim_texts, dense_top50, passages, cross_encoder, TOP_K)

    rankings = {"bm25": bm25, "dense": dense, "hybrid": hybrid, "dense_rerank_zero_shot": reranked}
    variants = {}
    for name, ranking in rankings.items():
        texts = {cid: [passages[e] for e in ids[:TOP_K]] for cid, ids in ranking.items()}
        variants[name] = verify(build_examples(dev_claims, texts, TOP_K))

    dense_texts = {cid: [passages[e] for e in ids[:TOP_K]] for cid, ids in dense.items()}
    conditions = evidence_conditions(dense_texts, misleading, evidence, seed, TOP_K)
    return {
        "variants": variants,
        "retrieval_top5": {
            cid: [[e, round(s, 4)] for e, s in ranked[:TOP_K]]
            for cid, ranked in dense_scored.items()
        },
        "gate_scores": {
            name: max_similarity(dense_model, claim_texts, texts)
            for name, texts in conditions.items()
        },
    }
