import json
import time
from datetime import date

from sentence_transformers import CrossEncoder, SentenceTransformer

from truthlens.averitec import gold_by_claim, load_averitec_tables
from truthlens.bm25_retrieval import retrieve as bm25_retrieve
from truthlens.config import PROJECT_ROOT
from truthlens.dense_retrieval import retrieve as dense_retrieve
from truthlens.hybrid_retrieval import fuse_rankings
from truthlens.rerank import rerank
from truthlens.retrieval_metrics import evaluate_retrieval

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEPTH = 100
TOP_K = 50
SHOWN = ["mrr", "recall@1", "recall@5", "recall@10", "recall@20", "recall@50"]


def main() -> None:
    claims, evidence = load_averitec_tables()
    dev = claims[claims["split"] == "dev"]
    gold = gold_by_claim(evidence)
    claim_texts = dict(zip(dev["claim_id"], dev["claim"]))
    passages = dict(zip(evidence["evidence_id"], evidence["text"]))

    bm25 = bm25_retrieve(dev, evidence, k=DEPTH)
    dense = dense_retrieve(dev, evidence, SentenceTransformer(DENSE_MODEL), k=DEPTH)
    hybrid = fuse_rankings(bm25, dense, top_n=TOP_K)
    dense_top = {claim_id: ids[:TOP_K] for claim_id, ids in dense.items()}

    encoder = CrossEncoder(RERANK_MODEL)
    start = time.time()
    systems = {
        "dense": dense_top,
        "dense+rerank": rerank(claim_texts, dense_top, passages, encoder, TOP_K),
        "hybrid": hybrid,
        "hybrid+rerank": rerank(claim_texts, hybrid, passages, encoder, TOP_K),
    }
    seconds = round(time.time() - start, 1)

    results = {name: evaluate_retrieval(rankings, gold) for name, rankings in systems.items()}
    print(f"rerank time (two runs): {seconds}s")
    print(f"{'system':<15}" + "".join(f"{name:>10}" for name in SHOWN))
    for name, metrics in results.items():
        print(f"{name:<15}" + "".join(f"{metrics[key]:>10.4f}" for key in SHOWN))

    record = {
        "experiment": "retrieval_rerank",
        "date": date.today().isoformat(),
        "dense_model": DENSE_MODEL,
        "reranker": RERANK_MODEL,
        "candidates": f"top {TOP_K}",
        "metrics": results,
    }
    (PROJECT_ROOT / "experiments").mkdir(exist_ok=True)
    (PROJECT_ROOT / "experiments" / "retrieval_rerank.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
