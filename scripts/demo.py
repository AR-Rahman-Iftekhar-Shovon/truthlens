import argparse
import os

import gradio as gr
import pandas as pd

from truthlens.article import select_claims, split_sentences
from truthlens.averitec import load_averitec_tables
from truthlens.claim_detector import load_detector, score_sentences
from truthlens.claim_pipeline import verify_claims
from truthlens.live_pipeline import load_live_verifier, verify_live_claim
from truthlens.report_card import format_live_card

CSS = """
body, .gradio-container {background: #f6f3ec !important; font-family: Georgia, serif;}
.gradio-container {max-width: 100% !important; padding: 0 24px !important;}
.tl-head h1 {font-size: 34px; margin: 0; letter-spacing: -0.5px;}
.tl-head p {color: #5a5446; margin: 4px 0 0; font-size: 15px;}
.tl-card {background: #fffdf8; border-top: 4px solid var(--verdict); padding: 18px 20px;
  margin: 14px 0; box-shadow: 0 1px 0 #d8d2c4;}
.tl-claim {font-size: 19px; font-style: italic; margin: 0 0 10px;}
.tl-stamp {display: inline-block; color: var(--verdict); border: 2px solid var(--verdict);
  padding: 2px 10px; font: 700 13px/1.6 Arial, sans-serif; text-transform: uppercase;
  letter-spacing: 1px; transform: rotate(-2deg); margin: 0 0 8px;}
.tl-summary {font-size: 16px; margin: 4px 0 12px;}
.tl-card h4 {font: 700 12px Arial, sans-serif; text-transform: uppercase; color: #5a5446;}
.tl-card ol {padding-left: 18px; font-size: 14px; color: #3b372e;}
.tl-card a {color: #3b372e;}
.tl-card details {font: 13px Arial, sans-serif; color: #5a5446;}
.tl-picked {font: 14px Arial, sans-serif; color: #5a5446;}
.tl-src a, .tl-tag {font: 12px Arial, sans-serif; color: #5a5446; margin-left: 6px;}
.tl-card li {margin-bottom: 8px;}
"""
INTRO = (
    '<div class="tl-head"><h1>TruthLens</h1><p>Checks a claim against evidence and says '
    "&ldquo;not enough evidence&rdquo; instead of guessing. <i>Search the web</i> reads "
    "trusted sites live; <i>Fact-check archive</i> uses 9,453 fact-check notes from "
    "2018&ndash;2021. A research demo, not a fact-checker.</p></div>"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--detector-seed", type=int, default=42)
    parser.add_argument("--top-n", type=int, default=3)
    args = parser.parse_args()

    claims, evidence = load_averitec_tables()
    detector = load_detector(args.detector_seed)

    def check(texts: list[str]) -> str:
        queries = pd.DataFrame({"claim_id": [f"d-{i}" for i in range(len(texts))], "claim": texts})
        return "".join(verify_claims(queries, claims, evidence, args.seed, html=True))

    live = {}

    def check_web(text: str) -> str:
        if not text.strip():
            return "<p>Type a claim first.</p>"
        if not os.getenv("TAVILY_API_KEY"):
            return "<p>Web search needs TAVILY_API_KEY in the environment.</p>"
        if not live:
            live.update(load_live_verifier(args.seed, "nli", "advanced", 10))
        return format_live_card(verify_live_claim(text.strip(), live))

    def check_claim(text: str) -> str:
        return check([text.strip()]) if text.strip() else "<p>Type a claim first.</p>"

    def check_article(text: str) -> str:
        sentences = split_sentences(text)
        if not sentences:
            return "<p>Paste an article first.</p>"
        chosen = select_claims(sentences, score_sentences(sentences, detector), args.top_n)
        if not chosen:
            return "<p>No sentence looked like a checkable claim.</p>"
        picked = (
            f'<p class="tl-picked">{len(sentences)} sentences read, {len(chosen)} checkable '
            "claims found:</p>"
        )
        return picked + check([c["claim"] for c in chosen])

    with gr.Blocks(css=CSS, title="TruthLens") as demo:
        gr.HTML(INTRO)
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Tab("Search the web"):
                    web = gr.Textbox(label="Claim", lines=3)
                    web_button = gr.Button("Search trusted sources", variant="primary")
                with gr.Tab("Fact-check archive"):
                    claim = gr.Textbox(label="Claim", lines=3)
                    claim_button = gr.Button("Check", variant="primary")
                    gr.Examples(CLAIM_EXAMPLES, inputs=claim)
                with gr.Tab("Article"):
                    article = gr.Textbox(label="Article text", lines=8)
                    article_button = gr.Button("Find and check claims", variant="primary")
                    gr.Examples([ARTICLE_EXAMPLE], inputs=article)
            with gr.Column(scale=3):
                report = gr.HTML("<p class='tl-picked'>The report appears here.</p>")
        web_button.click(check_web, web, report)
        claim_button.click(check_claim, claim, report)
        article_button.click(check_article, article, report)
    demo.launch(share=True)


if __name__ == "__main__":
    main()
