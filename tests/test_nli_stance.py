import numpy as np

from truthlens.nli_stance import reorder_nli_probs


def test_reorder_nli_probs_to_label_order():
    id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
    probs = np.array([[0.1, 0.7, 0.2]])
    reordered = reorder_nli_probs(probs, id2label)
    assert reordered.tolist() == [[0.7, 0.1, 0.2]]
