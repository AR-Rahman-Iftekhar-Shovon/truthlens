import numpy as np

from truthlens.decision import INSUFFICIENT, decide


def test_decide_abstains_on_low_confidence_low_score_and_unknown():
    probs = np.array([[0.9, 0.05, 0.05], [0.5, 0.4, 0.1], [0.1, 0.1, 0.8], [0.9, 0.05, 0.05]])
    scores = np.array([0.6, 0.6, 0.6, 0.1])

    decisions = decide(probs, scores, prob_threshold=0.6, score_threshold=0.3)

    assert decisions == ["SUPPORTED", INSUFFICIENT, INSUFFICIENT, INSUFFICIENT]
