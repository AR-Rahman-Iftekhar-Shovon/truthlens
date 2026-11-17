import argparse
import os
from pathlib import Path

import gradio as gr
import pandas as pd

from truthlens.archive_search import search_archive
from truthlens.article import select_claims, split_sentences
from truthlens.averitec import load_averitec_tables
from truthlens.claim_detector import load_detector, score_sentences
from truthlens.claim_pipeline import verify_claims
from truthlens.live_pipeline import load_live_verifier, verify_live_claim
from truthlens.report_card import format_archive_results, format_live_card

CSS = Path(__file__).with_name("demo_style.css").read_text(encoding="utf-8")
MODES = ["Search the web", "Check with archive", "Analyse an article", "Browse archive", "Method"]
HEADER = (
    '<div id="tl-header"><h1>Truth<b>Lens</b></h1>'
    "<p>Evidence first. A verdict only when the evidence is strong enough.</p>"
    '<span class="tl-badge">Experimental research demo</span></div>'
)
EMPTY = (
    '<div class="tl-card"><h3 class="tl-claim">Your report will appear here</h3>'
    '<p class="tl-summary">TruthLens finds evidence for a claim, judges the claim against it, '
    "and answers <b>Not enough evidence</b> instead of guessing.</p>"
    "<p class='tl-note'>Try an example on the left, or type your own claim.</p></div>"
)
SIDE = {
    "Search the web": (
        "How web search decides",
        [
            "Searches a short list of fact-checkers, news agencies and official sites.",
            "<b>False</b> needs a fact-checking site that rates the claim false.",
            "<b>Supported</b> needs two different trusted sites that agree.",
            "Otherwise: <b>Not enough evidence</b>. Missing evidence never means false.",
        ],
    ),
    "Check with archive": (
        "How the archive check works",
        [
            "Finds the 5 closest notes among 9,453 fact-check notes (2018&ndash;2021).",
            "A model compares the claim with those notes.",
            "It answers only at 75% confidence and a good evidence match.",
            "New or local news is usually not in the archive.",
        ],
    ),
    "Analyse an article": (
        "How article analysis works",
        [
            "Splits the article into sentences.",
            "A model picks up to 3 sentences that are checkable claims.",
            "Each claim is checked against the fact-check archive.",
        ],
    ),
    "Browse archive": (
        "About the archive",
        [
            "Real-world claims checked by fact-checkers (AVeriTeC dataset).",
            "Each claim has the original verdict, date and evidence notes with sources.",
        ],
    ),
    "Method": (
        "Honest limits",
        [
            "Evidence archive is closed and covers about 2018&ndash;2021.",
            "Archive check: about 84% correct on the claims it answers, answering about 42%.",
            "Web search answered 3 of 20 test claims, all 3 correctly.",
            "Not a production fact-checker.",
        ],
    ),
}
METHOD = (
    '<div class="tl-card"><h3 class="tl-claim">How TruthLens works</h3><ol class="tl-summary">'
    "<li>Find evidence: keyword search (BM25) and meaning search (MiniLM) combined.</li>"
    "<li>Judge: a DistilRoBERTa model reads the claim with the top 5 notes.</li>"
    "<li>Calibrate: confidence is adjusted so 80% means about 80% correct.</li>"
    "<li>Abstain: no verdict below 75% confidence or with weakly related evidence.</li>"
    "</ol><p class='tl-note'>Code, results and limitations are documented in the GitHub "
    "repository (README, docs/results.md, docs/limitations.md).</p></div>"
)
CLAIM_EXAMPLES = [
    "China seek for court’s approval to kill the over 20,000 coronavirus patients to avoid "
    "further spread of the virus.",
    "The Prime Minister of Bangladesh is named Shovon.",
]
ARTICLE_EXAMPLE = (
    "China seek for court’s approval to kill the over 20,000 coronavirus patients to avoid "
    "further spread of the virus. Nigeria’s urban population at the time of independence was "
    "approximately 7 million. Many visitors arrived at the airport early that morning and the "
    "weather was pleasant."
)


def side_html(mode: str) -> str:
    title, points = SIDE[mode]
    items = "".join(f"<li>{p}</li>" for p in points)
    return f'<div class="tl-side"><h3>{title}</h3><ol>{items}</ol></div>'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--detector-seed", type=int, default=42)
    parser.add_argument("--top-n", type=int, default=3)
    args = parser.parse_args()

    claims, evidence = load_averitec_tables()
    detector = load_detector(args.detector_seed)
    live = {}

    def check(texts: list[str]) -> str:
        queries = pd.DataFrame({"claim_id": [f"d-{i}" for i in range(len(texts))], "claim": texts})
        return "".join(verify_claims(queries, claims, evidence, args.seed, html=True))

    def check_web(text: str) -> str:
        if not text.strip():
            return EMPTY
        if not os.getenv("TAVILY_API_KEY"):
            return "<p class='tl-note'>Web search needs TAVILY_API_KEY in the environment.</p>"
        if not live:
            live.update(load_live_verifier(args.seed, "nli", "advanced", 10))
        return format_live_card(verify_live_claim(text.strip(), live))

    def check_claim(text: str) -> str:
        return check([text.strip()]) if text.strip() else EMPTY

    def check_article(text: str) -> str:
        sentences = split_sentences(text)
        if not sentences:
            return EMPTY
        chosen = select_claims(sentences, score_sentences(sentences, detector), args.top_n)
        if not chosen:
            return "<p class='tl-note'>No sentence looked like a checkable claim.</p>"
        found = (
            f"<p class='tl-note'>{len(sentences)} sentences read, {len(chosen)} checkable "
            "claims found.</p>"
        )
        return found + check([c["claim"] for c in chosen])

    def browse(text: str) -> str:
        return format_archive_results(text, search_archive(text, claims, evidence))

    def switch(mode: str):
        return [gr.update(visible=m == mode) for m in MODES] + [side_html(mode)]

    with gr.Blocks(css=CSS, title="TruthLens") as demo:
        gr.HTML(HEADER)
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=180):
                nav = gr.Radio(MODES, value=MODES[0], show_label=False, elem_id="tl-nav")
            with gr.Column(scale=4):
                groups = []
                with gr.Group(visible=True, elem_classes="tl-panel") as group:
                    web = gr.Textbox(label="Claim to search on the web", lines=2)
                    web_button = gr.Button("Search trusted sources", variant="primary")
                    gr.Examples(CLAIM_EXAMPLES, inputs=web)
                groups.append(group)
                with gr.Group(visible=False, elem_classes="tl-panel") as group:
                    claim = gr.Textbox(label="Claim to check against the archive", lines=2)
                    claim_button = gr.Button("Check claim", variant="primary")
                    gr.Examples(CLAIM_EXAMPLES, inputs=claim)
                groups.append(group)
                with gr.Group(visible=False, elem_classes="tl-panel") as group:
                    article = gr.Textbox(label="Article text", lines=6)
                    article_button = gr.Button("Find and check claims", variant="primary")
                    gr.Examples([ARTICLE_EXAMPLE], inputs=article)
                groups.append(group)
                with gr.Group(visible=False, elem_classes="tl-panel") as group:
                    query = gr.Textbox(label="Search archived fact-checks", lines=1)
                    query_button = gr.Button("Search archive", variant="primary")
                groups.append(group)
                with gr.Group(visible=False) as group:
                    gr.HTML(METHOD)
                groups.append(group)
                report = gr.HTML(EMPTY)
            with gr.Column(scale=2, min_width=240):
                side = gr.HTML(side_html(MODES[0]))
        nav.change(switch, nav, groups + [side])
        web_button.click(check_web, web, report)
        claim_button.click(check_claim, claim, report)
        article_button.click(check_article, article, report)
        query_button.click(browse, query, report)
        query.submit(browse, query, report)
    demo.launch(share=True)


if __name__ == "__main__":
    main()
