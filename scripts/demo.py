import argparse

import gradio as gr
import pandas as pd

from truthlens.article import select_claims, split_sentences
from truthlens.averitec import load_averitec_tables
from truthlens.claim_detector import load_detector, score_sentences
from truthlens.claim_pipeline import verify_claims

CSS = """
.gradio-container {max-width: 760px !important; background: #f6f3ec; font-family: Georgia, serif;}
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
"""
INTRO = (
    '<div class="tl-head"><h1>TruthLens</h1><p>Checks a claim against evidence and says '
    "&ldquo;not enough evidence&rdquo; instead of guessing. Evidence comes from a fixed "
    "collection of 9,453 fact-check notes (2018&ndash;2021), so new or local news usually "
    "has no answer. A research demo, not a fact-checker.</p></div>"
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
        with gr.Tab("Check a claim"):
            claim = gr.Textbox(label="Claim", lines=2)
            gr.Examples(CLAIM_EXAMPLES, inputs=claim)
            claim_out = gr.HTML()
            gr.Button("Check", variant="primary").click(check_claim, claim, claim_out)
        with gr.Tab("Check an article"):
            article = gr.Textbox(label="Article text", lines=7)
            gr.Examples([ARTICLE_EXAMPLE], inputs=article)
            article_out = gr.HTML()
            gr.Button("Find and check claims", variant="primary").click(
                check_article, article, article_out
            )
    demo.launch(share=True)


if __name__ == "__main__":
    main()
