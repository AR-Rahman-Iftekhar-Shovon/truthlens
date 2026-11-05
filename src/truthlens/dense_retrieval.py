import numpy as np
import pandas as pd


def embed(model, texts: list[str], batch_size: int = 128) -> np.ndarray:
    return model.encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )


def retrieve(
    claims: pd.DataFrame, evidence: pd.DataFrame, model, k: int = 50
) -> dict[str, list[str]]:
    evidence_vectors = embed(model, evidence["text"].tolist())
    claim_vectors = embed(model, claims["claim"].tolist())
    scores = claim_vectors @ evidence_vectors.T
    evidence_ids = evidence["evidence_id"].to_numpy()

    rankings = {}
    for claim_id, row in zip(claims["claim_id"], scores):
        top = np.argsort(-row, kind="stable")[:k]
        rankings[claim_id] = evidence_ids[top].tolist()
    return rankings
