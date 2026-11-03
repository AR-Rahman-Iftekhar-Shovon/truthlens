import json

import pandas as pd

from truthlens.averitec_corpus import build_tables
from truthlens.config import PROJECT_ROOT, load_config


def main() -> None:
    config = load_config()
    raw_dir = PROJECT_ROOT / config["paths"]["data_raw"] / "averitec"
    out_dir = PROJECT_ROOT / config["paths"]["data_processed"] / "averitec"
    out_dir.mkdir(parents=True, exist_ok=True)

    claim_tables, evidence_tables = [], []
    for split in ("train", "dev"):
        claims = json.loads((raw_dir / f"{split}.json").read_text(encoding="utf-8"))
        claim_df, evidence_df = build_tables(claims, split)
        claim_tables.append(claim_df)
        evidence_tables.append(evidence_df)

    claims_df = pd.concat(claim_tables, ignore_index=True)
    evidence_df = pd.concat(evidence_tables, ignore_index=True)
    claims_df.to_csv(out_dir / "claims.csv", index=False, lineterminator="\n")
    evidence_df.to_csv(out_dir / "evidence.csv", index=False, lineterminator="\n")

    print(f"claims: {len(claims_df)}, evidence passages: {len(evidence_df)}")
    for split, part in claims_df.groupby("split"):
        no_evidence = int((part["n_evidence"] == 0).sum())
        print(f"{split}: {len(part)} claims, {no_evidence} without any answerable evidence")
    print(claims_df.groupby(["split", "label_mapped"]).size().to_string())


if __name__ == "__main__":
    main()
