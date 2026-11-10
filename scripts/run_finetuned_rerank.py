import json
from datetime import date

from sentence_transformers import SentenceTransformer

from truthlens.averitec import gold_by_claim, load_averitec_tables
from truthlens.bm25_retrieval import retrieve as bm25_retrieve
from truthlens.config import PROJECT_ROOT
from truthlens.dense_retrieval import retrieve as dense_retrieve
from truthlens.hybrid_retrieval import fuse_rankings
from truthlens.rerank import rerank
from truthlens.rerank_data import build_training_pairs
from truthlens.rerank_train import fine_tune_reranker
from truthlens.retrieval_metrics import evaluate_retrieval
from truthlens.transformer_train import set_torch_seed

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SEED = 42
EPOCHS = 2
LR = 2e-5
BATCH_SIZE = 32
DEPTH = 100
TOP_K = 50
SHOWN = ["mrr", "recall@1", "recall@5", "recall@10", "recall@20", "recall@50"]


def main() -> None:
    set_torch_seed(SEED)
    claims, evidence = load_averitec_tables()
    gold = gold_by_claim(evidence)
    claim_texts = dict(zip(claims["claim_id"], claims["claim"]))
    passages = dict(zip(evidence["evidence_id"], evidence["text"]))
    dense_model = SentenceTransformer(DENSE_MODEL)

    train_claims = claims[claims["split"] == "train"]
    train_evidence = evidence[evidence["claim_id"].isin(set(train_claims["claim_id"]))]
    train_rankings = dense_retrieve(train_claims, train_evidence, dense_model, k=20)
    pairs, labels = build_training_pairs(claim_texts, passages, gold, train_rankings)
    print(f"training pairs: {len(pairs)}, positives: {sum(labels)}")

    scorer = fine_tune_reranker(RERANK_MODEL, pairs, labels, EPOCHS, LR, BATCH_SIZE, SEED)

    dev = claims[claims["split"] == "dev"]
    bm25 = bm25_retrieve(dev, evidence, k=DEPTH)
    dense = dense_retrieve(dev, evidence, dense_model, k=DEPTH)
    hybrid = fuse_rankings(bm25, dense, top_n=TOP_K)
    dense_top = {claim_id: ids[:TOP_K] for claim_id, ids in dense.items()}

    systems = {
        "dense": dense_top,
        "dense+ft_rerank": rerank(claim_texts, dense_top, passages, scorer, TOP_K),
        "hybrid": hybrid,
        "hybrid+ft_rerank": rerank(claim_texts, hybrid, passages, scorer, TOP_K),
    }
    results = {name: evaluate_retrieval(rankings, gold) for name, rankings in systems.items()}

    print(f"{'system':<18}" + "".join(f"{name:>10}" for name in SHOWN))
    for name, metrics in results.items():
        print(f"{name:<18}" + "".join(f"{metrics[key]:>10.4f}" for key in SHOWN))

    record = {
        "experiment": "retrieval_finetuned_rerank",
        "date": date.today().isoformat(),
        "reranker": RERANK_MODEL,
        "training": {"epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE, "seed": SEED},
        "training_pairs": len(pairs),
        "metrics": results,
    }
    (PROJECT_ROOT / "experiments").mkdir(exist_ok=True)
    (PROJECT_ROOT / "experiments" / "retrieval_finetuned_rerank.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
