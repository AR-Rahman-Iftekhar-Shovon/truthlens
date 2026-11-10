from truthlens.averitec import load_averitec_tables
from truthlens.config import PROJECT_ROOT, load_config
from truthlens.misleading import build_misleading_set


def main() -> None:
    claims, evidence = load_averitec_tables()
    misleading = build_misleading_set(claims, evidence)

    out_dir = PROJECT_ROOT / load_config()["paths"]["data_processed"] / "averitec"
    misleading.to_csv(out_dir / "misleading_dev.csv", index=False, lineterminator="\n")

    covered = misleading["query_claim_id"].nunique()
    opposite = (misleading["query_label"] != misleading["neighbor_label"]).mean()
    print(f"rows: {len(misleading)}, dev claims covered: {covered}/500")
    print(f"rows with different label than the query claim: {opposite:.3f}")
    print(f"mean similarity: {misleading['similarity'].mean():.3f}")
    for _, row in misleading.sample(3, random_state=42).iterrows():
        print(f"\nQUERY ({row['query_label']}): {row['query_claim']}")
        print(f"NEIGHBOR ({row['neighbor_label']}): {row['neighbor_claim']}")
        print(f"EVIDENCE: {row['text'][:200]}")


if __name__ == "__main__":
    main()
