import argparse
from pathlib import Path

import pandas as pd

from truthlens.article import select_claims, split_sentences
from truthlens.averitec import load_averitec_tables
from truthlens.claim_detector import load_detector, score_sentences
from truthlens.claim_pipeline import verify_claims


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=44, help="verifier seed")
    parser.add_argument("--detector-seed", type=int, default=42)
    args = parser.parse_args()

    sentences = split_sentences(Path(args.file).read_text(encoding="utf-8"))
    scores = score_sentences(sentences, load_detector(args.detector_seed))
    chosen = select_claims(sentences, scores, args.top_n, args.min_score)
    print(f"{len(sentences)} sentences, {len(chosen)} check-worthy claims selected\n")
    if not chosen:
        print("No sentence reached the check-worthy threshold.")
        return
    for item in chosen:
        print(f"[check-worthy {item['score']:.2f}] {item['claim']}")
    print("-" * 60)

    claims, evidence = load_averitec_tables()
    ids = [f"a-{i}" for i in range(len(chosen))]
    queries = pd.DataFrame({"claim_id": ids, "claim": [c["claim"] for c in chosen]})
    for report in verify_claims(queries, claims, evidence, args.seed):
        print(report + "\n" + "-" * 60)


if __name__ == "__main__":
    main()
