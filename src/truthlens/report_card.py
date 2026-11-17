from html import escape

VERDICTS = {
    "SUPPORTED": ("Supported", "The evidence we found agrees with this claim.", "#1d6b3a"),
    "REFUTED": ("False", "The evidence we found contradicts this claim.", "#a3241b"),
    "INSUFFICIENT_EVIDENCE": ("Not enough evidence", "We could not judge this claim.", "#6b6455"),
}


def format_card(
    claim: str,
    assessment: str,
    confidence: float,
    probabilities: dict[str, float],
    evidence: list[dict],
    similarity: float,
    reason: str | None,
) -> str:
    title, summary, color = VERDICTS[assessment]
    if reason:
        summary += f" Reason: {escape(reason)}."
    sources = "".join(
        f'<li><a href="{escape(item["source_url"])}" target="_blank">{escape(item["text"][:160])}'
        "</a></li>"
        if item["source_url"]
        else f"<li>{escape(item['text'][:160])}</li>"
        for item in evidence
    )
    numbers = ", ".join(f"{label.lower()} {value:.0%}" for label, value in probabilities.items())
    return (
        f'<article class="tl-card" style="--verdict:{color}">'
        f'<p class="tl-claim">“{escape(claim)}”</p>'
        f'<p class="tl-stamp">{title}</p>'
        f'<p class="tl-summary">{summary}</p>'
        f"<h4>What we read</h4><ol>{sources}</ol>"
        f"<details><summary>Model details</summary><p>Model confidence: {numbers}. "
        f"Best evidence match: {similarity:.2f} (needs 0.55). Answers need 75% confidence.</p>"
        f"</details></article>"
    )
