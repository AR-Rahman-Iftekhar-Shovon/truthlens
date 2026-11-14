CONFIDENCE_THRESHOLD = 0.75


def error_category(
    truth: str,
    guess: str,
    confidence: float,
    gold_hits: int,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> str:
    if truth == guess:
        return "correct"
    if gold_hits == 0:
        return "retrieval_failure"
    if "UNKNOWN" in (truth, guess):
        return "insufficient_vs_verdict"
    return "overconfident_flip" if confidence >= threshold else "low_confidence_flip"
