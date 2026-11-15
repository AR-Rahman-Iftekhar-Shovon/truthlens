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


def load_live_verifier(seed: int) -> dict:
    outputs = PROJECT_ROOT / "outputs"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(LABELS))
    model.load_state_dict(torch.load(outputs / f"verifier_seed{seed}.pt"))
    device = torch.device("cuda")
    model.to(device)
    probs_file = json.loads((outputs / f"verifier_probs_seed{seed}.json").read_text())
    val_probs, val_labels, _ = load_rows(probs_file["val_retrieved"])
    return {
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "dense_model": SentenceTransformer(DENSE_MODEL),
        "temperature": fit_temperature(val_probs, val_labels),
        "domains": load_trusted_domains(),
        "fact_checkers": load_fact_checkers(),
    }


def verify_live_claim(claim: str, verifier: dict) -> dict:
    snippets = search_evidence(claim, verifier["domains"])
    if not snippets:
        return {
            "claim": claim,
            "verdict": "INSUFFICIENT_EVIDENCE",
            "reason": "no results from trusted domains",
            "stances": [],
        }
    tokenizer = verifier["tokenizer"]
    texts = [s["snippet"] for s in snippets]
    dataset = StatementDataset([claim] * len(texts), [0] * len(texts), tokenizer, MAX_LENGTH, texts)
    loader = DataLoader(dataset, batch_size=16, collate_fn=DataCollatorWithPadding(tokenizer))
    probabilities = np.array(predict_proba(verifier["model"], loader, verifier["device"]))
    calibrated = softmax_with_temperature(probabilities, verifier["temperature"])

    dense_model = verifier["dense_model"]
    claim_vector = dense_model.encode([claim], normalize_embeddings=True)[0]
    similarities = (dense_model.encode(texts, normalize_embeddings=True) @ claim_vector).tolist()

    stances = build_stances(
        snippets, calibrated.tolist(), similarities, LABELS, SIMILARITY_THRESHOLD
    )
    result = decide_live_verdict(stances, verifier["fact_checkers"])
    return {"claim": claim, "stances": stances, **result}
