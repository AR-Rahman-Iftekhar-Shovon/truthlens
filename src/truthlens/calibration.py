import numpy as np


def softmax_with_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-6, 1.0)) / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray) -> float:
    chosen = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(chosen, 1e-12, 1.0))))


def fit_temperature(probs: np.ndarray, labels: np.ndarray) -> float:
    grid = np.arange(0.5, 5.01, 0.05)
    losses = [negative_log_likelihood(softmax_with_temperature(probs, t), labels) for t in grid]
    return float(grid[int(np.argmin(losses))])


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def reliability_table(probs: np.ndarray, labels: np.ndarray, bins: int = 5) -> list[tuple]:
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.any():
            rows.append(
                (
                    round(float(low), 1),
                    round(float(high), 1),
                    int(in_bin.sum()),
                    round(float(confidence[in_bin].mean()), 3),
                    round(float(correct[in_bin].mean()), 3),
                )
            )
    return rows
