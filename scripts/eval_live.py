import argparse
import json

from truthlens.config import PROJECT_ROOT
from truthlens.live_eval import score_live
from truthlens.live_pipeline import SIMILARITY_THRESHOLD, load_live_verifier, verify_live_claim
from truthlens.live_verdict import MIN_DOMAINS, MIN_STANCE_CONFIDENCE

CLAIMS_FILE = PROJECT_ROOT / "experiments" / "custom_eval_claims.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--stance", choices=["verifier", "nli"], default="verifier")
    args = parser.parse_args()

    claims = json.loads(CLAIMS_FILE.read_text(encoding="utf-8"))
    verifier = load_live_verifier(args.seed, args.stance)
    rows = []
    for item in claims:
        result = verify_live_claim(item["claim"], verifier)
        rows.append({"id": item["id"], "label": item["label"], **result})
        print(f"{item['id']} gold={item['label']} pred={result['verdict']} ({result['reason']})")

    summary = score_live(rows)
    print(json.dumps(summary, indent=2))

    outputs = PROJECT_ROOT / "outputs"
    outputs.mkdir(exist_ok=True)
    (outputs / f"live_eval_predictions_{args.stance}.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    settings = {
        "stance_model": args.stance,
        "seed": args.seed,
        "similarity_gate": SIMILARITY_THRESHOLD,
        "stance_confidence": MIN_STANCE_CONFIDENCE,
        "min_domains": MIN_DOMAINS,
    }
    (PROJECT_ROOT / "experiments" / f"live_eval_{args.stance}.json").write_text(
        json.dumps({"settings": settings, **summary}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
