import json
from datetime import date

from truthlens.averitec import gold_by_claim, load_averitec_tables
from truthlens.bm25_retrieval import retrieve
from truthlens.config import PROJECT_ROOT
from truthlens.retrieval_metrics import evaluate_retrieval


def main() -> None:
    claims, evidence = load_averitec_tables()
    dev = claims[claims["split"] == "dev"]

    rankings = retrieve(dev, evidence)
    metrics = evaluate_retrieval(rankings, gold_by_claim(evidence))
    for name, value in metrics.items():
        print(f"{name}: {value}")

    (PROJECT_ROOT / "outputs").mkdir(exist_ok=True)
    (PROJECT_ROOT / "outputs" / "rankings_bm25.json").write_text(json.dumps(rankings))

    result = {
        "experiment": "retrieval_bm25",
        "date": date.today().isoformat(),
        "queries": "500 dev claims",
        "pool": f"{len(evidence)} evidence passages (train + dev)",
        "tokenizer": "lowercase, [a-z0-9]+, no stop words",
        "bm25": "rank_bm25.BM25Okapi default parameters",
        "metrics": metrics,
    }
    (PROJECT_ROOT / "experiments").mkdir(exist_ok=True)
    (PROJECT_ROOT / "experiments" / "retrieval_bm25.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
