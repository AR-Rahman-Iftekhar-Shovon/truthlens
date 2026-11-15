import argparse

from truthlens.live_search import load_trusted_domains, search_evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim", required=True)
    args = parser.parse_args()

    results = search_evidence(args.claim, load_trusted_domains())
    if not results:
        print("No evidence from trusted domains (this does not mean the claim is false).")
    for i, item in enumerate(results, 1):
        print(f"\n{i}. {item['title']}\n{item['url']}\n{item['snippet'][:500]}")


if __name__ == "__main__":
    main()
