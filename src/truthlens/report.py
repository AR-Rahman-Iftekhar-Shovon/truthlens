def abstention_reason(
    label: str,
    confidence: float,
    similarity: float,
    confidence_threshold: float,
    similarity_threshold: float,
) -> str | None:
    if similarity < similarity_threshold:
        return (
            f"the retrieved evidence is only weakly related to the claim "
            f"(similarity {similarity:.2f} < {similarity_threshold})"
        )
    if label == "UNKNOWN":
        return "the verifier judged the evidence insufficient for a verdict"
    if confidence < confidence_threshold:
        return f"confidence {confidence:.2f} is below the threshold {confidence_threshold}"
    return None


def uncertainty_level(abstained: bool, confidence: float) -> str:
    if abstained or confidence < 0.6:
        return "High"
    return "Medium" if confidence < 0.8 else "Low"


def format_report(
    claim: str,
    assessment: str,
    confidence: float,
    probabilities: dict[str, float],
    evidence: list[dict],
    similarity: float,
    reason: str | None,
) -> str:
    lines = [
        f"Assessment: {assessment}",
        f"Confidence: {confidence:.2f}",
        f"Claim: {claim}",
        "Evidence:",
    ]
    for number, item in enumerate(evidence, start=1):
        source = item["source_url"] or "no source url"
        lines.append(f"  {number}. {item['text'][:220]} [{source}]")
    shown = ", ".join(f"{label} {value:.2f}" for label, value in probabilities.items())
    lines.append(f"Verifier probabilities (calibrated): {shown}")
    if reason:
        lines.append(f"Why: no verdict because {reason}.")
    else:
        lines.append(f"Why: the verifier favours {assessment} given the retrieved evidence.")
    lines.append(f"Evidence quality: highest claim-evidence similarity {similarity:.2f}")
    lines.append(f"Uncertainty: {uncertainty_level(reason is not None, confidence)}")
    return "\n".join(lines)
