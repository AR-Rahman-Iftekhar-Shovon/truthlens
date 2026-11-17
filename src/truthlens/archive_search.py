import numpy as np
import pandas as pd

from truthlens.bm25_retrieval import build_index, tokenize


def search_archive(
    query: str, claims: pd.DataFrame, evidence: pd.DataFrame, k: int = 8
) -> list[dict]:
    """Archived fact-checked claims closest to the query (BM25 over claim text)."""
    rows = claims[claims["label_mapped"] != ""].reset_index(drop=True)
    index = build_index(rows.rename(columns={"claim": "text"}))
    terms = set(tokenize(query))
    scores = index.get_scores(list(terms))
    ranked = np.argsort(-scores, kind="stable")
    top = [i for i in ranked if terms & set(tokenize(rows.iloc[i]["claim"]))][:k]
    notes = evidence.groupby("claim_id")
    results = []
    for i in top:
        row = rows.iloc[i]
        group = notes.get_group(row["claim_id"]) if row["claim_id"] in notes.groups else None
        results.append(
            {
                "claim": row["claim"],
                "label": row["label"],
                "date": row["claim_date"],
                "notes": []
                if group is None
                else group[["question", "answer", "source_url"]].to_dict("records"),
            }
        )
    return results
