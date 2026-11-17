import json
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from truthlens.analysis_dump import max_similarity
from truthlens.bm25_retrieval import retrieve as bm25_retrieve
from truthlens.calibration import fit_temperature, softmax_with_temperature
from truthlens.config import PROJECT_ROOT
from truthlens.decision import INSUFFICIENT, load_rows
from truthlens.dense_retrieval import cached_embeddings
from truthlens.dense_retrieval import retrieve as dense_retrieve
from truthlens.device import pick_device
from truthlens.hybrid_retrieval import fuse_rankings
from truthlens.report import abstention_reason, format_report
from truthlens.report_card import format_card
from truthlens.transformer_data import StatementDataset
from truthlens.transformer_train import predict_proba
from truthlens.verification_data import LABELS, format_evidence

MODEL = "distilroberta-base"
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CONFIDENCE_THRESHOLD = 0.75
SIMILARITY_THRESHOLD = 0.55
TOP_K = 5
DEPTH = 100
MAX_LENGTH = 384


@lru_cache(maxsize=1)
def load_dense_model() -> SentenceTransformer:
    return SentenceTransformer(DENSE_MODEL)


@lru_cache(maxsize=1)
def load_verifier(seed: int) -> tuple:
    outputs = PROJECT_ROOT / "outputs"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(LABELS))
    device = pick_device()
    model.load_state_dict(torch.load(outputs / f"verifier_seed{seed}.pt", map_location=device))
    model.to(device)
    probs_file = json.loads((outputs / f"verifier_probs_seed{seed}.json").read_text())
    val_probs, val_labels, _ = load_rows(probs_file["val_retrieved"])
    return tokenizer, model, device, fit_temperature(val_probs, val_labels)


def verify_claims(
    queries: pd.DataFrame,
    claims: pd.DataFrame,
    evidence: pd.DataFrame,
    seed: int,
    html: bool = False,
) -> list[str]:
    """One report per query claim."""
    dense_model = load_dense_model()
    bm25 = bm25_retrieve(queries, evidence, k=DEPTH)
    vectors = cached_embeddings(
        dense_model, evidence["text"].tolist(), PROJECT_ROOT / "outputs" / "evidence_embeddings.npy"
    )
    dense = dense_retrieve(queries, evidence, dense_model, k=DEPTH, evidence_vectors=vectors)
    hybrid = fuse_rankings(bm25, dense, top_n=TOP_K)

    by_id = evidence.set_index("evidence_id")
    texts = {cid: [by_id.loc[e, "text"] for e in ids] for cid, ids in hybrid.items()}
    claim_texts = dict(zip(queries["claim_id"], queries["claim"]))
    gold_labels = dict(zip(claims["claim_id"], claims["label_mapped"]))
    similarity = max_similarity(dense_model, claim_texts, texts)

    tokenizer, model, device, temperature = load_verifier(seed)
    claim_list = queries["claim"].tolist()
    evidence_list = [format_evidence(texts[cid], TOP_K) for cid in queries["claim_id"]]
    dataset = StatementDataset(
        claim_list, [0] * len(claim_list), tokenizer, MAX_LENGTH, evidence_list
    )
    loader = DataLoader(dataset, batch_size=16, collate_fn=DataCollatorWithPadding(tokenizer))
    probabilities = np.array(predict_proba(model, loader, device))
    calibrated = softmax_with_temperature(probabilities, temperature)

    reports = []
    for index, claim_id in enumerate(queries["claim_id"]):
        probability = calibrated[index]
        label = LABELS[int(probability.argmax())]
        confidence = float(probability.max())
        score = similarity[claim_id]
        reason = abstention_reason(
            label, confidence, score, CONFIDENCE_THRESHOLD, SIMILARITY_THRESHOLD
        )
        items = [
            {"text": text, "source_url": by_id.loc[e, "source_url"]}
            for e, text in zip(hybrid[claim_id], texts[claim_id])
        ]
        render = format_card if html else format_report
        report = render(
            claim_texts[claim_id],
            INSUFFICIENT if reason else label,
            confidence,
            dict(zip(LABELS, probability)),
            items,
            score,
            reason,
        )
        if claim_id in gold_labels and not html:
            report += f"\nGold label (dev): {gold_labels[claim_id]}"
        reports.append(report)
    return reports
