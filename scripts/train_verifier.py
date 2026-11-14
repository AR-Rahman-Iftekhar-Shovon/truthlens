import argparse
import json
from datetime import date

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from truthlens.analysis_dump import collect
from truthlens.averitec import load_averitec_tables, load_misleading_passages
from truthlens.config import PROJECT_ROOT
from truthlens.dense_retrieval import retrieve_with_scores
from truthlens.evaluation import summarize
from truthlens.robustness import build_conditions
from truthlens.transformer_data import StatementDataset
from truthlens.transformer_train import fine_tune, predict, predict_proba, set_torch_seed
from truthlens.verification_data import LABELS, build_examples

MODEL = "distilroberta-base"
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_LENGTH = 384
TOP_K = 5
EPOCHS = 4
LR = 2e-5
BATCH_SIZE = 16


def make_dataset(examples, tokenizer, with_evidence: bool) -> StatementDataset:
    labels = [LABELS.index(label) for label in examples["label"]]
    pair = examples["evidence"].tolist() if with_evidence else None
    return StatementDataset(examples["claim"].tolist(), labels, tokenizer, MAX_LENGTH, pair)


def make_loader(dataset, tokenizer) -> DataLoader:
    return DataLoader(dataset, batch_size=64, collate_fn=DataCollatorWithPadding(tokenizer))


def evaluate(model, dataset, tokenizer, device) -> dict:
    predicted = [LABELS[i] for i in predict(model, make_loader(dataset, tokenizer), device)]
    return summarize([LABELS[i] for i in dataset.labels], predicted, LABELS)


def report(name: str, summary: dict) -> dict:
    f1 = {label: round(summary["per_class"][label]["f1-score"], 3) for label in LABELS}
    print(f"{name:<36} macro_f1={summary['macro_f1']} acc={summary['accuracy']} f1={f1}")
    print(f"{'':<36} confusion={summary['confusion_matrix']}")
    return {"macro_f1": summary["macro_f1"], "accuracy": summary["accuracy"], "f1": f1}


def dump(examples, dataset, model, tokenizer, device, scores=None) -> list[dict]:
    probabilities = predict_proba(model, make_loader(dataset, tokenizer), device)
    rows = []
    for claim_id, label, probs in zip(examples["claim_id"], examples["label"], probabilities):
        row = {"claim_id": claim_id, "label": label, "probs": [round(p, 5) for p in probs]}
        if scores is not None:
            row.update(scores[claim_id])
        rows.append(row)
    return rows


def retrieve_evidence(claims_subset, evidence, passages, dense_model):
    scored = retrieve_with_scores(claims_subset, evidence, dense_model, k=TOP_K)
    texts = {cid: [passages[e] for e, _ in ranked] for cid, ranked in scored.items()}
    scores = {
        cid: {
            "top1_score": round(ranked[0][1], 4),
            "mean_score": round(float(np.mean([s for _, s in ranked])), 4),
        }
        for cid, ranked in scored.items()
    }
    return texts, scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--robustness", action="store_true")
    parser.add_argument("--analysis", action="store_true")
    args = parser.parse_args()
    seed = args.seed
    set_torch_seed(seed)
    device = torch.device("cuda")
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    claims, evidence = load_averitec_tables()
    passages = dict(zip(evidence["evidence_id"], evidence["text"]))
    gold_texts = evidence.groupby("claim_id")["text"].apply(list).to_dict()
    dense_model = SentenceTransformer(DENSE_MODEL)
    dev_claims = claims[claims["split"] == "dev"]

    train_all = build_examples(claims[claims["split"] == "train"], gold_texts, TOP_K)
    shuffled = train_all.sample(frac=1.0, random_state=seed)
    n_val = int(0.15 * len(shuffled))
    val, train = shuffled.iloc[:n_val], shuffled.iloc[n_val:]
    val_claims = claims[claims["claim_id"].isin(set(val["claim_id"]))]

    val_texts, val_scores = retrieve_evidence(val_claims, evidence, passages, dense_model)
    dev_texts, dev_scores = retrieve_evidence(dev_claims, evidence, passages, dense_model)
    val_retrieved = build_examples(val_claims, val_texts, TOP_K)
    dev_gold = build_examples(dev_claims, gold_texts, TOP_K)
    dev_retrieved = build_examples(dev_claims, dev_texts, TOP_K)
    print(f"train {len(train)}, val {len(val)}, dev {len(dev_gold)}")

    counts = train["label"].value_counts()
    weights = [1 / np.sqrt(counts[label]) for label in LABELS]
    weights = [w / np.mean(weights) for w in weights]

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    id_to_label = dict(enumerate(LABELS))
    results = {}
    for variant, with_evidence in (("claim_only", False), ("claim+gold_evidence", True)):
        model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(LABELS))
        _, _, best_state = fine_tune(
            model,
            make_dataset(train, tokenizer, with_evidence),
            make_dataset(val, tokenizer, with_evidence),
            tokenizer,
            id_to_label,
            EPOCHS,
            LR,
            BATCH_SIZE,
            seed,
            class_weights=weights,
        )
        model.load_state_dict(best_state)
        model.to(device)

        if not with_evidence:
            summary = evaluate(model, make_dataset(dev_gold, tokenizer, False), tokenizer, device)
            results["dev, claim only"] = report(f"{variant} | dev", summary)
            continue

        gold_set = make_dataset(dev_gold, tokenizer, True)
        retrieved_set = make_dataset(dev_retrieved, tokenizer, True)
        val_set = make_dataset(val_retrieved, tokenizer, True)
        summary = evaluate(model, gold_set, tokenizer, device)
        results["dev, gold evidence"] = report(f"{variant} | dev gold evidence", summary)
        summary = evaluate(model, retrieved_set, tokenizer, device)
        results["dev, retrieved evidence"] = report(f"{variant} | dev top-{TOP_K}", summary)
        if args.robustness:
            conditions = build_conditions(
                dev_claims, dev_texts, load_misleading_passages(), evidence, seed
            )
            robustness = {}
            for name, examples in conditions.items():
                dataset = make_dataset(examples, tokenizer, True)
                robustness[name] = dump(examples, dataset, model, tokenizer, device)
            (out_dir / f"verifier_robustness_seed{seed}.json").write_text(json.dumps(robustness))

        if args.analysis:
            analysis = collect(
                dev_claims,
                evidence,
                passages,
                dense_model,
                CrossEncoder(RERANK_MODEL),
                load_misleading_passages(),
                seed,
                lambda examples: dump(
                    examples, make_dataset(examples, tokenizer, True), model, tokenizer, device
                ),
            )
            (out_dir / f"verifier_analysis_seed{seed}.json").write_text(json.dumps(analysis))
        probabilities = {
            "labels": LABELS,
            "val_retrieved": dump(val_retrieved, val_set, model, tokenizer, device, val_scores),
            "dev_retrieved": dump(
                dev_retrieved, retrieved_set, model, tokenizer, device, dev_scores
            ),
            "dev_gold": dump(dev_gold, gold_set, model, tokenizer, device),
        }
        torch.save(best_state, out_dir / f"verifier_seed{seed}.pt")
        (out_dir / f"verifier_probs_seed{seed}.json").write_text(json.dumps(probabilities))

    record = {
        "experiment": "verification",
        "date": date.today().isoformat(),
        "seed": seed,
        "model": MODEL,
        "max_length": MAX_LENGTH,
        "top_k": TOP_K,
        "results": results,
    }
    (PROJECT_ROOT / "experiments").mkdir(exist_ok=True)
    path = PROJECT_ROOT / "experiments" / f"verification_seed{seed}.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
