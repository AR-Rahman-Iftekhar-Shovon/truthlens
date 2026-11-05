import argparse
import json
import time
from datetime import date

from sentence_transformers import SentenceTransformer

from truthlens.averitec import gold_by_claim, load_averitec_tables
from truthlens.config import PROJECT_ROOT
from truthlens.dense_retrieval import retrieve
from truthlens.retrieval_metrics import evaluate_retrieval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    claims, evidence = load_averitec_tables()
    dev = claims[claims["split"] == "dev"]
    model = SentenceTransformer(args.model)

    start = time.time()
    rankings = retrieve(dev, evidence, model)
    seconds = round(time.time() - start, 1)
    metrics = evaluate_retrieval(rankings, gold_by_claim(evidence))

    print(f"retrieval time: {seconds}s")
    for name, value in metrics.items():
        print(f"{name}: {value}")

    short_name = args.model.split("/")[-1].lower().replace("-", "_")
    result = {
        "experiment": f"retrieval_dense_{short_name}",
        "date": date.today().isoformat(),
        "queries": "500 dev claims",
        "pool": f"{len(evidence)} evidence passages (train + dev)",
        "model": args.model,
        "seconds": seconds,
        "metrics": metrics,
    }
    (PROJECT_ROOT / "experiments").mkdir(exist_ok=True)
    (PROJECT_ROOT / "experiments" / f"retrieval_dense_{short_name}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
