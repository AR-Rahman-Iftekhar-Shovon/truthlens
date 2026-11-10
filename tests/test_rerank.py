from truthlens.rerank import rerank


class FakeCrossEncoder:
    def predict(self, pairs, **kwargs):
        return [len(set(query.split()) & set(passage.split())) for query, passage in pairs]


def test_rerank_orders_by_model_score():
    result = rerank(
        {"c1": "moon landing"},
        {"c1": ["e1", "e2"]},
        {"e1": "tax rates", "e2": "moon landing fake"},
        FakeCrossEncoder(),
    )

    assert result["c1"] == ["e2", "e1"]
