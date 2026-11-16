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
from truthlens.nli_stance import load_nli_stance_fn
from truthlens.transformer_data import StatementDataset
from truthlens.transformer_train import predict_proba
from truthlens.verification_data import LABELS

MODEL = "distilroberta-base"
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.55
MAX_LENGTH = 384


def verifier_probabilities_fn(seed: int, device: torch.device):
    outputs = PROJECT_ROOT / "outputs"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(LABELS))
    model.load_state_dict(torch.load(outputs / f"verifier_seed{seed}.pt"))
    model.to(device)
    probs_file = json.loads((outputs / f"verifier_probs_seed{seed}.json").read_text())
    val_probs, val_labels, _ = load_rows(probs_file["val_retrieved"])
    temperature = fit_temperature(val_probs, val_labels)

    def probabilities(claims: list[str], evidence: list[str]) -> np.ndarray:
        dataset = StatementDataset(claims, [0] * len(claims), tokenizer, MAX_LENGTH, evidence)
        loader = DataLoader(dataset, batch_size=16, collate_fn=DataCollatorWithPadding(tokenizer))
        raw = np.array(predict_proba(model, loader, device))
        return softmax_with_temperature(raw, temperature)

    return probabilities


def load_live_verifier(
    seed: int, stance: str = "verifier", depth: str = "basic", max_results: int = 5
) -> dict:
    device = torch.device("cuda")
    if stance == "verifier":
        probabilities = verifier_probabilities_fn(seed, device)

        def stance_fn(claim: str, snippets: list[str]) -> np.ndarray:
            return probabilities([claim] * len(snippets), snippets)

    else:
        stance_fn = load_nli_stance_fn(device)
    return {
        "stance_fn": stance_fn,
        "dense_model": SentenceTransformer(DENSE_MODEL),
        "domains": load_trusted_domains(),
        "fact_checkers": load_fact_checkers(),
        "depth": depth,
        "max_results": max_results,
    }


def verify_live_claim(claim: str, verifier: dict) -> dict:
    snippets = search_evidence(
        claim, verifier["domains"], verifier["max_results"], depth=verifier["depth"]
    )
    if not snippets:
        return {
            "claim": claim,
            "verdict": "INSUFFICIENT_EVIDENCE",
            "reason": "no results from trusted domains",
            "stances": [],
        }
    texts = [s["snippet"] for s in snippets]
    probabilities = verifier["stance_fn"](claim, texts)

    dense_model = verifier["dense_model"]
    claim_vector = dense_model.encode([claim], normalize_embeddings=True)[0]
    similarities = (dense_model.encode(texts, normalize_embeddings=True) @ claim_vector).tolist()

    stances = build_stances(
        snippets, probabilities.tolist(), similarities, LABELS, SIMILARITY_THRESHOLD
    )
    result = decide_live_verdict(stances, verifier["fact_checkers"])
    return {"claim": claim, "stances": stances, **result}
