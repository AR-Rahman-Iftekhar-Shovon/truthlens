import argparse

import gradio as gr
import pandas as pd

from truthlens.article import select_claims, split_sentences
from truthlens.averitec import load_averitec_tables
from truthlens.claim_detector import load_detector, score_sentences
from truthlens.claim_pipeline import verify_claims

NOTE = (
    "Evidence comes from a closed pool of 9453 passages (about 2018-2021), so new or local "
    "claims usually end as INSUFFICIENT_EVIDENCE. This is a research demo, not a fact-checker."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--detector-seed", type=int, default=42)
    parser.add_argument("--top-n", type=int, default=3)
    args = parser.parse_args()

    claims, evidence = load_averitec_tables()
    detector = load_detector(args.detector_seed)

    def run(text: str, mode: str) -> str:
        text = text.strip()
        if not text:
            return "Please enter a claim or an article."
        header = ""
        if mode == "Article":
            sentences = split_sentences(text)
            chosen = select_claims(sentences, score_sentences(sentences, detector), args.top_n)
            if not chosen:
                return "No sentence reached the check-worthy threshold."
            header = "Selected claims:\n" + "\n".join(
                f"  [{c['score']:.2f}] {c['claim']}" for c in chosen
            )
            texts = [c["claim"] for c in chosen]
        else:
            texts = [text]
        queries = pd.DataFrame({"claim_id": [f"d-{i}" for i in range(len(texts))], "claim": texts})
        reports = verify_claims(queries, claims, evidence, args.seed)
        return "\n\n".join(([header] if header else []) + reports)

    demo = gr.Interface(
        fn=run,
        inputs=[
            gr.Textbox(lines=6, label="Claim or article"),
            gr.Radio(["Claim", "Article"], value="Claim", label="Input type"),
        ],
        outputs=gr.Textbox(lines=28, label="Report"),
        title="TruthLens",
        description=NOTE,
    )
    demo.launch(share=True)


if __name__ == "__main__":
    main()
