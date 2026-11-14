import numpy as np

from truthlens.decision import INSUFFICIENT, decide, select_operating_point


def test_decide_abstains_on_low_confidence_low_score_and_unknown():
    probs = np.array([[0.9, 0.05, 0.05], [0.5, 0.4, 0.1], [0.1, 0.1, 0.8], [0.9, 0.05, 0.05]])
    scores = np.array([0.6, 0.6, 0.6, 0.1])

    decisions = decide(probs, scores, prob_threshold=0.6, score_threshold=0.3)

    assert decisions == ["SUPPORTED", INSUFFICIENT, INSUFFICIENT, INSUFFICIENT]


def test_select_operating_point_meets_target_accuracy():
    probs = np.array([[0.9, 0.05, 0.05], [0.6, 0.3, 0.1], [0.1, 0.8, 0.1], [0.4, 0.5, 0.1]])
    labels = np.array([0, 1, 1, 0])
    scores = np.array([0.5, 0.5, 0.5, 0.5])

    chosen = select_operating_point(probs, labels, scores, 0.9, [0.0, 0.7, 0.85], [0.0])

    assert chosen == (0.5, 0.7, 0.0)
