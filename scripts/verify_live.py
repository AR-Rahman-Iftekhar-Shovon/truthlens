import argparse

from truthlens.live_pipeline import load_live_verifier, verify_live_claim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim", action="append", required=True)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--stance", choices=["verifier", "nli"], default="verifier")
    args = parser.parse_args()

    verifier = load_live_verifier(args.seed, args.stance)
    for claim in args.claim:
        result = verify_live_claim(claim, verifier)
        print(f"Claim: {claim}")
        for s in result["stances"]:
            print(f"  [{s['stance']} {s['confidence']:.2f} sim {s['similarity']:.2f}] {s['url']}")
        print(f"Verdict: {result['verdict']} ({result['reason']})\n" + "-" * 60)


if __name__ == "__main__":
    main()
