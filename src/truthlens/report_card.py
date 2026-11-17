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
    sources = "".join(_source_line(item["text"], item["source_url"]) for item in evidence)
    numbers = ", ".join(f"{label.lower()} {value:.0%}" for label, value in probabilities.items())
    return (
        f'<article class="tl-card" style="--verdict:{color}">'
        f'<p class="tl-claim">“{escape(claim)}”</p>'
        f'<p class="tl-stamp">{title}</p>'
        f'<p class="tl-summary">{summary}</p>'
        f"<h4>Closest fact-check notes we found</h4><ol>{sources}</ol>"
        f"<details><summary>Model details</summary><p>Model confidence: {numbers}. "
        f"Best evidence match: {similarity:.2f} (needs 0.55). Answers need 75% confidence.</p>"
        f"</details></article>"
    )


def _domain(url: str) -> str:
    return url.split("//")[-1].split("/")[0].removeprefix("www.")


def _source_line(text: str, url: str) -> str:
    question, _, answer = text.removeprefix("Q: ").partition(" A: ")
    body = f"<b>{escape(question)}</b><br>{escape(answer[:220])}" if answer else escape(text[:220])
    link = f' <a href="{escape(url)}" target="_blank">{escape(_domain(url))}</a>' if url else ""
    return f"<li>{body}<span class='tl-src'>{link}</span></li>"


def format_live_card(result: dict) -> str:
    title, summary, color = VERDICTS[result["verdict"]]
    used = [s for s in result["stances"] if s["stance"] != "UNKNOWN"]
    items = "".join(
        f"<li><span class='tl-tag'>{s['stance'].lower()} {s['confidence']:.0%}</span> "
        f"{escape(s['snippet'][:200])} "
        f'<a href="{escape(s["url"])}" target="_blank">{escape(s["domain"])}</a></li>'
        for s in used
    )
    read = len(result["stances"])
    return (
        f'<article class="tl-card" style="--verdict:{color}">'
        f'<p class="tl-claim">“{escape(result["claim"])}”</p>'
        f'<p class="tl-stamp">{title}</p>'
        f'<p class="tl-summary">{summary} Reason: {escape(result["reason"])}.</p>'
        f"<h4>Trusted sources that took a side ({len(used)} of {read} read)</h4>"
        f"<ol>{items or '<li>None of the results clearly agreed or disagreed.</li>'}</ol>"
        "</article>"
    )
