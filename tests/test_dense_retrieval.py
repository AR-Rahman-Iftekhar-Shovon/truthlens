import numpy as np
import pandas as pd

from truthlens.dense_retrieval import retrieve, retrieve_with_scores


class FakeModel:
    vectors = {
        "moon claim": [1.0, 0.0],
        "moon evidence": [0.9, 0.1],
        "tax evidence": [0.0, 1.0],
    }

    def encode(self, texts, **kwargs):
        return np.array([self.vectors[text] for text in texts])


def test_retrieve_orders_by_similarity():
    claims = pd.DataFrame({"claim_id": ["c1"], "claim": ["moon claim"]})
    evidence = pd.DataFrame(
        {"evidence_id": ["e1", "e2"], "text": ["tax evidence", "moon evidence"]}
    )

    assert retrieve(claims, evidence, FakeModel(), k=2)["c1"] == ["e2", "e1"]


def test_retrieve_with_scores_returns_descending_scores():
    claims = pd.DataFrame({"claim_id": ["c1"], "claim": ["moon claim"]})
    evidence = pd.DataFrame(
        {"evidence_id": ["e1", "e2"], "text": ["tax evidence", "moon evidence"]}
    )

    ranked = retrieve_with_scores(claims, evidence, FakeModel(), k=2)["c1"]

    assert [evidence_id for evidence_id, _ in ranked] == ["e2", "e1"]
    assert ranked[0][1] > ranked[1][1]
