import numpy as np
import pandas as pd

from truthlens.dense_retrieval import retrieve


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
