MIN_DOMAINS = 2
MIN_STANCE_CONFIDENCE = 0.75


def decide_live_verdict(
    snippets: list[dict],
    fact_checkers: set[str],
    min_domains: int = MIN_DOMAINS,
    min_confidence: float = MIN_STANCE_CONFIDENCE,
) -> dict:
    """Combine per-snippet stances (domain, stance, confidence) into one verdict.

    No coverage or conflicting evidence gives INSUFFICIENT_EVIDENCE, never REFUTED.
    """
    strong = [s for s in snippets if s["confidence"] >= min_confidence]
    supporting = {s["domain"] for s in strong if s["stance"] == "SUPPORTED"}
    refuting = {s["domain"] for s in strong if s["stance"] == "REFUTED"}

    if supporting and refuting:
        return _result("INSUFFICIENT_EVIDENCE", "trusted sources disagree", supporting, refuting)
    if refuting & fact_checkers:
        return _result(
            "REFUTED", "a fact-checking site rates the claim false", supporting, refuting
        )
    if len(supporting) >= min_domains:
        return _result(
            "SUPPORTED", f"{len(supporting)} independent trusted domains", supporting, refuting
        )
    if len(refuting) >= min_domains:
        return _result(
            "REFUTED", f"{len(refuting)} independent trusted domains", supporting, refuting
        )
    return _result(
        "INSUFFICIENT_EVIDENCE", "not enough independent trusted evidence", supporting, refuting
    )


def _result(verdict: str, reason: str, supporting: set[str], refuting: set[str]) -> dict:
    return {
        "verdict": verdict,
        "reason": reason,
        "supporting_domains": sorted(supporting),
        "refuting_domains": sorted(refuting),
    }


def build_stances(
    snippets: list[dict],
    probabilities: list[list[float]],
    similarities: list[float],
    labels: list[str],
    min_similarity: float,
) -> list[dict]:
    """Attach a stance per snippet; snippets unrelated to the claim count as UNKNOWN."""
    stances = []
    for snippet, probs, similarity in zip(snippets, probabilities, similarities, strict=True):
        best = max(range(len(labels)), key=lambda i: probs[i])
        related = similarity >= min_similarity
        stances.append(
            {
                **snippet,
                "stance": labels[best] if related else "UNKNOWN",
                "confidence": float(probs[best]),
                "similarity": float(similarity),
            }
        )
    return stances
