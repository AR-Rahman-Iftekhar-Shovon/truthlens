import numpy as np

from truthlens.verification_data import LABELS

INSUFFICIENT = "INSUFFICIENT_EVIDENCE"


def decide(
    probs: np.ndarray, top1_scores: np.ndarray, prob_threshold: float, score_threshold: float
) -> list[str]:
    decisions = []
    for prob, score in zip(probs, top1_scores):
        label = LABELS[int(prob.argmax())]
        if label == "UNKNOWN" or prob.max() < prob_threshold or score < score_threshold:
            decisions.append(INSUFFICIENT)
        else:
            decisions.append(label)
    return decisions
