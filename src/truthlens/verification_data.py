import pandas as pd

LABELS = ["SUPPORTED", "REFUTED", "UNKNOWN"]


def format_evidence(passages: list[str], max_passages: int = 5) -> str:
    return " ".join(passages[:max_passages])


def build_examples(
    claims: pd.DataFrame, evidence_by_claim: dict[str, list[str]], max_passages: int = 5
) -> pd.DataFrame:
    rows = claims[claims["label_mapped"].isin(LABELS)]
    return pd.DataFrame(
        {
            "claim_id": rows["claim_id"].to_numpy(),
            "claim": rows["claim"].to_numpy(),
            "label": rows["label_mapped"].to_numpy(),
            "evidence": [
                format_evidence(evidence_by_claim.get(claim_id, []), max_passages)
                for claim_id in rows["claim_id"]
            ],
        }
    )
