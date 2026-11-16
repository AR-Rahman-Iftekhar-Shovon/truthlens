import argparse

import pandas as pd

from truthlens.averitec import load_averitec_tables
from truthlens.claim_pipeline import verify_claims


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim", action="append")
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()

    claims, evidence = load_averitec_tables()
    if args.claim:
        ids = [f"q-{i}" for i in range(len(args.claim))]
        queries = pd.DataFrame({"claim_id": ids, "claim": args.claim})
    else:
        queries = claims[claims["split"] == "dev"].sample(3, random_state=7)[["claim_id", "claim"]]
        print("No --claim given: 3 dev claims (their own evidence is inside the closed corpus).\n")
    verify_claims(queries, claims, evidence, args.seed)


if __name__ == "__main__":
    main()
