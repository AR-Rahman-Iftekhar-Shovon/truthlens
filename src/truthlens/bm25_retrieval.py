import re

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def build_index(evidence: pd.DataFrame) -> BM25Okapi:
    return BM25Okapi([tokenize(text) for text in evidence["text"]])


def retrieve(claims: pd.DataFrame, evidence: pd.DataFrame, k: int = 50) -> dict[str, list[str]]:
    index = build_index(evidence)
    evidence_ids = evidence["evidence_id"].to_numpy()

    rankings = {}
    for claim_id, claim_text in zip(claims["claim_id"], claims["claim"]):
        scores = index.get_scores(tokenize(claim_text))
        top = np.argsort(-scores, kind="stable")[:k]
        rankings[claim_id] = evidence_ids[top].tolist()
    return rankings
