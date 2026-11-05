import json
from datetime import date

from sentence_transformers import SentenceTransformer

from truthlens.averitec import gold_by_claim, load_averitec_tables
from truthlens.bm25_retrieval import retrieve as bm25_retrieve
from truthlens.config import PROJECT_ROOT
from truthlens.dense_retrieval import retrieve as dense_retrieve
from truthlens.hybrid_retrieval import fuse_rankings
from truthlens.retrieval_metrics import evaluate_retrieval

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEPTH = 100
TOP_K = 50
SHOWN = ["mrr", "recall@1", "recall@5", "recall@10", "recall@20", "recall@50"]


def coverage(bm25: dict, dense: dict, gold: dict) -> dict[str, float]:
    counts = {"only_bm25": 0, "only_dense": 0, "both": 0, "neither": 0}
    for claim_id in bm25:
        found_bm25 = set(bm25[claim_id][:TOP_K])
        found_dense = set(dense[claim_id][:TOP_K])
        for evidence_id in gold[claim_id]:
            in_bm25, in_dense = evidence_id in found_bm25, evidence_id in found_dense
            key = (
                "both"
                if in_bm25 and in_dense
                else "only_bm25"
                if in_bm25
                else ("only_dense" if in_dense else "neither")
            )
            counts[key] += 1
    total = sum(counts.values())
    return {name: round(count / total, 4) for name, count in counts.items()}


def main() -> None:
    claims, evidence = load_averitec_tables()
    dev = claims[claims["split"] == "dev"]
    gold = gold_by_claim(evidence)

    bm25 = bm25_retrieve(dev, evidence, k=DEPTH)
    dense = dense_retrieve(dev, evidence, SentenceTransformer(MODEL), k=DEPTH)
    hybrid = fuse_rankings(bm25, dense, top_n=TOP_K)

    systems = {"bm25": bm25, "dense": dense, "hybrid_rrf": hybrid}
    results = {name: evaluate_retrieval(rankings, gold) for name, rankings in systems.items()}

    print(f"{'system':<12}" + "".join(f"{name:>10}" for name in SHOWN))
    for name, metrics in results.items():
        print(f"{name:<12}" + "".join(f"{metrics[key]:>10.4f}" for key in SHOWN))

    overlap = coverage(bm25, dense, gold)
    print(f"\ngold passages found in top {TOP_K}: {overlap}")

    result = {
        "experiment": "retrieval_hybrid_rrf",
        "date": date.today().isoformat(),
        "model": MODEL,
        "fusion": "reciprocal rank fusion, k=60, depth 100 per system",
        "metrics": results,
        "coverage_top50": overlap,
    }
    (PROJECT_ROOT / "experiments").mkdir(exist_ok=True)
    (PROJECT_ROOT / "experiments" / "retrieval_hybrid_rrf.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
