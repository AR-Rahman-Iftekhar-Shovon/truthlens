import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

OPPOSITE = {"SUPPORTED": "REFUTED", "REFUTED": "SUPPORTED"}


def build_misleading_set(
    claims: pd.DataFrame,
    evidence: pd.DataFrame,
    query_split: str = "dev",
    neighbors: int = 3,
    per_neighbor: int = 2,
    min_similarity: float = 0.1,
    max_similarity: float = 0.8,
) -> pd.DataFrame:
    matrix = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(claims["claim"])
    query_positions = np.flatnonzero((claims["split"] == query_split).to_numpy())
    similarity = linear_kernel(matrix[query_positions], matrix)
    evidence_by_claim = {claim_id: group for claim_id, group in evidence.groupby("claim_id")}

    rows = []
    for row_index, position in enumerate(query_positions):
        query = claims.iloc[position]
        scores = similarity[row_index].copy()
        scores[position] = -1.0
        scores[(scores < min_similarity) | (scores > max_similarity)] = -1.0
        wanted = OPPOSITE.get(query["label_mapped"])

        chosen = 0
        for candidate in np.argsort(-scores, kind="stable"):
            if scores[candidate] < 0 or chosen == neighbors:
                break
            neighbor = claims.iloc[candidate]
            if wanted and neighbor["label_mapped"] != wanted:
                continue
            for _, passage in evidence_by_claim[neighbor["claim_id"]].head(per_neighbor).iterrows():
                rows.append(
                    {
                        "query_claim_id": query["claim_id"],
                        "query_claim": query["claim"],
                        "query_label": query["label_mapped"],
                        "neighbor_claim_id": neighbor["claim_id"],
                        "neighbor_claim": neighbor["claim"],
                        "neighbor_label": neighbor["label_mapped"],
                        "similarity": round(float(scores[candidate]), 4),
                        "evidence_id": passage["evidence_id"],
                        "text": passage["text"],
                    }
                )
            chosen += 1
    return pd.DataFrame(rows)
