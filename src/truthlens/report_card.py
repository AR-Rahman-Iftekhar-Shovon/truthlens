from html import escape

VERDICTS = {
    "SUPPORTED": ("Supported", "The evidence we found agrees with this claim.", "supported"),
    "REFUTED": ("False", "The evidence we found contradicts this claim.", "refuted"),
    "INSUFFICIENT_EVIDENCE": (
        "Not enough evidence",
        "We could not judge this claim.",
        "unknown",
    ),
}


def _domain(url: str) -> str:
    return url.split("//")[-1].split("/")[0].removeprefix("www.")


def _link(url: str) -> str:
    if not url:
        return ""
    return f'<a class="tl-src" href="{escape(url)}" target="_blank">{escape(_domain(url))} ↗</a>'


def _meter(label: str, value: float) -> str:
    return (
        f'<div class="tl-meter"><span>{label}</span><div class="tl-bar">'
        f'<i style="width:{value:.0%}"></i></div><b>{value:.0%}</b></div>'
    )


def _header(claim: str, assessment: str, summary: str) -> str:
    title, default, css = VERDICTS[assessment]
    return (
        f'<article class="tl-card tl-{css}"><div class="tl-top">'
        f'<span class="tl-pill">{title}</span></div>'
        f'<h3 class="tl-claim">{escape(claim)}</h3>'
        f'<p class="tl-summary">{summary or default}</p>'
    )


def _source_line(text: str, url: str) -> str:
    question, _, answer = text.removeprefix("Q: ").partition(" A: ")
    body = (
        f"<b>{escape(question)}</b><span>{escape(answer[:240])}</span>"
        if answer
        else f"<span>{escape(text[:240])}</span>"
    )
    return f'<li class="tl-ev">{body}{_link(url)}</li>'


def format_card(
    claim: str,
    assessment: str,
    confidence: float,
    probabilities: dict[str, float],
    evidence: list[dict],
    similarity: float,
    reason: str | None,
) -> str:
    _, default, _ = VERDICTS[assessment]
    summary = default + (f" Reason: {escape(reason)}." if reason else "")
    sources = "".join(_source_line(item["text"], item["source_url"]) for item in evidence)
    meters = "".join(_meter(label.title(), value) for label, value in probabilities.items())
    return (
        _header(claim, assessment, summary)
        + _meter("Evidence match", max(similarity, 0.0))
        + f"<details><summary>Model confidence</summary>{meters}"
        "<p class='tl-note'>An answer needs 75% confidence and an evidence match of 0.55.</p>"
        "</details>"
        + f"<h4>Closest fact-check notes</h4><ul class='tl-list'>{sources}</ul></article>"
    )


def format_live_card(result: dict) -> str:
    _, default, _ = VERDICTS[result["verdict"]]
    summary = f"{default} Reason: {escape(result['reason'])}."
    groups = []
    for stance, heading in (("SUPPORTED", "Supporting"), ("REFUTED", "Contradicting")):
        items = [s for s in result["stances"] if s["stance"] == stance]
        rows = "".join(
            f'<li class="tl-ev"><span>{escape(s["snippet"][:240])}</span>'
            f"<em>{s['confidence']:.0%} sure</em>{_link(s['url'])}</li>"
            for s in items
        )
        groups.append(
            f'<h4 class="tl-{stance.lower()}-h">{heading} ({len(items)})</h4>'
            f"<ul class='tl-list'>{rows or '<li class=tl-empty>None</li>'}</ul>"
        )
    used = sum(s["stance"] != "UNKNOWN" for s in result["stances"])
    read = len(result["stances"])
    return (
        _header(result["claim"], result["verdict"], summary)
        + f"<p class='tl-note'>{used} of {read} trusted results took a side.</p>"
        + "".join(groups)
        + "</article>"
    )


def format_archive_results(query: str, results: list[dict]) -> str:
    if not results:
        return f"<p class='tl-note'>No archived fact-check matches “{escape(query)}”.</p>"
    blocks = []
    for item in results:
        notes = "".join(
            _source_line(f"Q: {n['question']} A: {n['answer']}", n["source_url"])
            for n in item["notes"]
        )
        blocks.append(
            f'<article class="tl-card tl-archive"><div class="tl-top">'
            f'<span class="tl-pill">{escape(item["label"])}</span>'
            f'<span class="tl-date">{escape(item["date"])}</span></div>'
            f'<h3 class="tl-claim">{escape(item["claim"])}</h3>'
            f"<details><summary>{len(item['notes'])} evidence notes</summary>"
            f"<ul class='tl-list'>{notes}</ul></details></article>"
        )
    return "".join(blocks)
