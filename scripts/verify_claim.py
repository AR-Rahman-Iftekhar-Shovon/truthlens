import argparse
import json

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from truthlens.analysis_dump import max_similarity
from truthlens.averitec import load_averitec_tables
from truthlens.bm25_retrieval import retrieve as bm25_retrieve
from truthlens.calibration import fit_temperature, softmax_with_temperature
from truthlens.config import PROJECT_ROOT
from truthlens.decision import INSUFFICIENT, load_rows
from truthlens.dense_retrieval import retrieve as dense_retrieve
from truthlens.hybrid_retrieval import fuse_rankings
from truthlens.report import abstention_reason, format_report
from truthlens.transformer_data import StatementDataset
from truthlens.transformer_train import predict_proba
from truthlens.verification_data import LABELS, format_evidence

MODEL = "distilroberta-base"
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CONFIDENCE_THRESHOLD = 0.75
SIMILARITY_THRESHOLD = 0.45
TOP_K = 5
DEPTH = 100
MAX_LENGTH = 384


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim", action="append")
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()

    claims, evidence = load_averitec_tables()
    if args.claim:
        ids = [f"q-{i}" for i in range(len(args.claim))]
        queries = pd.DataFrame({"claim_id": ids, "claim": args.claim})
    else:
        queries = claims[claims["split"] == "dev"].sample(3, random_state=7)[["claim_id", "claim"]]
        print("No --claim given: 3 dev claims (their own evidence is inside the closed corpus).\n")

    dense_model = SentenceTransformer(DENSE_MODEL)
    bm25 = bm25_retrieve(queries, evidence, k=DEPTH)
    dense = dense_retrieve(queries, evidence, dense_model, k=DEPTH)
    hybrid = fuse_rankings(bm25, dense, top_n=TOP_K)

    by_id = evidence.set_index("evidence_id")
    texts = {cid: [by_id.loc[e, "text"] for e in ids] for cid, ids in hybrid.items()}
    claim_texts = dict(zip(queries["claim_id"], queries["claim"]))
    similarity = max_similarity(dense_model, claim_texts, texts)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(LABELS))
    outputs = PROJECT_ROOT / "outputs"
    model.load_state_dict(torch.load(outputs / f"verifier_seed{args.seed}.pt"))
    device = torch.device("cuda")
    model.to(device)

    claim_list = queries["claim"].tolist()
    evidence_list = [format_evidence(texts[cid], TOP_K) for cid in queries["claim_id"]]
    dataset = StatementDataset(
        claim_list, [0] * len(claim_list), tokenizer, MAX_LENGTH, evidence_list
    )
    loader = DataLoader(dataset, batch_size=16, collate_fn=DataCollatorWithPadding(tokenizer))
    probabilities = np.array(predict_proba(model, loader, device))

    probs_file = json.loads((outputs / f"verifier_probs_seed{args.seed}.json").read_text())
    val_probs, val_labels, _ = load_rows(probs_file["val_retrieved"])
    temperature = fit_temperature(val_probs, val_labels)
    calibrated = softmax_with_temperature(probabilities, temperature)
    print(f"temperature {temperature:.2f}\n")

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
        print(
            format_report(
                claim_texts[claim_id],
                INSUFFICIENT if reason else label,
                confidence,
                dict(zip(LABELS, probability)),
                items,
                score,
                reason,
            )
        )
        print("-" * 60)


if __name__ == "__main__":
    main()
