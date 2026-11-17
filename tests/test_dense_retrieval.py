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


class CountingModel:
    def __init__(self):
        self.calls = 0

    def encode(self, texts, **kwargs):
        self.calls += 1
        return np.ones((len(texts), 2), dtype=np.float32)


def test_cached_embeddings_reuses_file(tmp_path):
    from truthlens.dense_retrieval import cached_embeddings

    model, path = CountingModel(), tmp_path / "vectors.npy"
    first = cached_embeddings(model, ["a", "b"], path)
    second = cached_embeddings(model, ["a", "b"], path)
    assert model.calls == 1 and first.shape == second.shape == (2, 2)
    cached_embeddings(model, ["a", "b", "c"], path)
    assert model.calls == 2
