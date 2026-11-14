import numpy as np

from truthlens.verification_data import LABELS

INSUFFICIENT = "INSUFFICIENT_EVIDENCE"


def load_rows(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs = np.array([row["probs"] for row in rows])
    labels = np.array([LABELS.index(row["label"]) for row in rows])
    scores = np.array([row.get("top1_score", 0.0) for row in rows])
    return probs, labels, scores


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


def coverage_and_accuracy(
    probs: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    prob_threshold: float,
    score_threshold: float,
) -> tuple[float, float]:
    decisions = decide(probs, scores, prob_threshold, score_threshold)
    answered = [decision != INSUFFICIENT for decision in decisions]
    hits = [d == LABELS[y] for d, y, a in zip(decisions, labels, answered) if a]
    return float(np.mean(answered)), float(np.mean(hits)) if hits else 0.0


def select_operating_point(
    probs: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    target_accuracy: float,
    prob_grid,
    score_grid,
) -> tuple[float, float, float] | None:
    best = None
    for prob_threshold in prob_grid:
        for score_threshold in score_grid:
            coverage, accuracy = coverage_and_accuracy(
                probs, labels, scores, prob_threshold, score_threshold
            )
            if accuracy >= target_accuracy and (best is None or coverage > best[0]):
                best = (coverage, float(prob_threshold), float(score_threshold))
    return best
