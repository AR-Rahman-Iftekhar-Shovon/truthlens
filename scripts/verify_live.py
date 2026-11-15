import argparse
import json

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from truthlens.calibration import fit_temperature, softmax_with_temperature
from truthlens.config import PROJECT_ROOT
from truthlens.decision import load_rows
from truthlens.live_search import load_fact_checkers, load_trusted_domains, search_evidence
from truthlens.live_verdict import build_stances, decide_live_verdict
from truthlens.transformer_data import StatementDataset
from truthlens.transformer_train import predict_proba
from truthlens.verification_data import LABELS

MODEL = "distilroberta-base"
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.55
MAX_LENGTH = 384


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim", action="append", required=True)
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()

    outputs = PROJECT_ROOT / "outputs"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(LABELS))
    model.load_state_dict(torch.load(outputs / f"verifier_seed{args.seed}.pt"))
    device = torch.device("cuda")
    model.to(device)
    dense_model = SentenceTransformer(DENSE_MODEL)

    probs_file = json.loads((outputs / f"verifier_probs_seed{args.seed}.json").read_text())
    val_probs, val_labels, _ = load_rows(probs_file["val_retrieved"])
    temperature = fit_temperature(val_probs, val_labels)

    domains = load_trusted_domains()
    fact_checkers = load_fact_checkers()

    for claim in args.claim:
        snippets = search_evidence(claim, domains)
        print(f"Claim: {claim}")
        if not snippets:
            print("Verdict: INSUFFICIENT_EVIDENCE (no results from trusted domains)\n" + "-" * 60)
            continue

        dataset = StatementDataset(
            [claim] * len(snippets),
            [0] * len(snippets),
            tokenizer,
            MAX_LENGTH,
            [s["snippet"] for s in snippets],
        )
        loader = DataLoader(dataset, batch_size=16, collate_fn=DataCollatorWithPadding(tokenizer))
        calibrated = softmax_with_temperature(
            np.array(predict_proba(model, loader, device)), temperature
        )

        claim_vector = dense_model.encode([claim], normalize_embeddings=True)[0]
        snippet_vectors = dense_model.encode(
            [s["snippet"] for s in snippets], normalize_embeddings=True
        )
        similarities = (snippet_vectors @ claim_vector).tolist()

        stances = build_stances(
            snippets, calibrated.tolist(), similarities, LABELS, SIMILARITY_THRESHOLD
        )
        result = decide_live_verdict(stances, fact_checkers)
        for s in stances:
            print(f"  [{s['stance']} {s['confidence']:.2f} sim {s['similarity']:.2f}] {s['url']}")
        print(f"Verdict: {result['verdict']} ({result['reason']})\n" + "-" * 60)


if __name__ == "__main__":
    main()
