from collections import Counter

ABSTAIN = "INSUFFICIENT_EVIDENCE"


def score_live(rows: list[dict]) -> dict:
    """rows: dicts with gold `label` and predicted `verdict`."""
    n = len(rows)
    answered = [r for r in rows if r["verdict"] != ABSTAIN]
    correct = [r for r in rows if r["verdict"] == r["label"]]
    wrong_answers = [r for r in answered if r["verdict"] != r["label"]]
    by_label = {}
    for label in sorted({r["label"] for r in rows}):
        group = [r for r in rows if r["label"] == label]
        by_label[label] = {
            "n": len(group),
            "correct": sum(r["verdict"] == label for r in group),
            "predicted": dict(Counter(r["verdict"] for r in group)),
        }
    return {
        "n": n,
        "accuracy": len(correct) / n if n else 0.0,
        "coverage": len(answered) / n if n else 0.0,
        "accuracy_on_answered": (len(answered) - len(wrong_answers)) / len(answered)
        if answered
        else None,
        "wrong_answers": len(wrong_answers),
        "by_label": by_label,
    }
