import pandas as pd

from truthlens.config import PROJECT_ROOT, load_config


def load_averitec_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = PROJECT_ROOT / load_config()["paths"]["data_processed"] / "averitec"
    claims = pd.read_csv(out_dir / "claims.csv", keep_default_na=False)
    evidence = pd.read_csv(out_dir / "evidence.csv", keep_default_na=False)
    return claims, evidence


def gold_by_claim(evidence: pd.DataFrame) -> dict[str, set[str]]:
    return evidence.groupby("claim_id")["evidence_id"].apply(set).to_dict()


def load_misleading_passages() -> dict[str, list[str]]:
    path = (
        PROJECT_ROOT / load_config()["paths"]["data_processed"] / "averitec" / "misleading_dev.csv"
    )
    frame = pd.read_csv(path, keep_default_na=False)
    return frame.groupby("query_claim_id")["text"].apply(list).to_dict()
