import argparse
import json
from datetime import date

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from truthlens.averitec import load_averitec_tables
from truthlens.config import PROJECT_ROOT
from truthlens.dense_retrieval import retrieve as dense_retrieve
from truthlens.evaluation import summarize
from truthlens.transformer_data import StatementDataset
from truthlens.transformer_train import fine_tune, predict, set_torch_seed
from truthlens.verification_data import LABELS, build_examples

MODEL = "distilroberta-base"
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LENGTH = 384
TOP_K = 5
EPOCHS = 4
LR = 2e-5
BATCH_SIZE = 16


def make_dataset(examples, tokenizer, with_evidence: bool) -> StatementDataset:
    labels = [LABELS.index(label) for label in examples["label"]]
    pair = examples["evidence"].tolist() if with_evidence else None
    return StatementDataset(examples["claim"].tolist(), labels, tokenizer, MAX_LENGTH, pair)


def evaluate(model, dataset, tokenizer, device) -> dict:
    loader = DataLoader(dataset, batch_size=64, collate_fn=DataCollatorWithPadding(tokenizer))
    predicted = [LABELS[i] for i in predict(model, loader, device)]
    return summarize([LABELS[i] for i in dataset.labels], predicted, LABELS)


def report(name: str, summary: dict) -> dict:
    f1_by_class = {label: round(summary["per_class"][label]["f1-score"], 3) for label in LABELS}
    print(f"{name:<34} macro_f1={summary['macro_f1']}  acc={summary['accuracy']}  f1={f1_by_class}")
    print(f"{'':<34} confusion={summary['confusion_matrix']}")
    return {"macro_f1": summary["macro_f1"], "accuracy": summary["accuracy"], "f1": f1_by_class}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    seed = parser.parse_args().seed
    set_torch_seed(seed)
    device = torch.device("cuda")

    claims, evidence = load_averitec_tables()
    passages = dict(zip(evidence["evidence_id"], evidence["text"]))
    gold_texts = evidence.groupby("claim_id")["text"].apply(list).to_dict()

    dev_claims = claims[claims["split"] == "dev"]
    dense = dense_retrieve(dev_claims, evidence, SentenceTransformer(DENSE_MODEL), k=TOP_K)
    retrieved_texts = {cid: [passages[e] for e in ids] for cid, ids in dense.items()}

    train_all = build_examples(claims[claims["split"] == "train"], gold_texts, TOP_K)
    shuffled = train_all.sample(frac=1.0, random_state=seed)
    n_val = int(0.15 * len(shuffled))
    val, train = shuffled.iloc[:n_val], shuffled.iloc[n_val:]
    dev_gold = build_examples(dev_claims, gold_texts, TOP_K)
    dev_retrieved = build_examples(dev_claims, retrieved_texts, TOP_K)
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

        if with_evidence:
            gold_set = make_dataset(dev_gold, tokenizer, True)
            retrieved_set = make_dataset(dev_retrieved, tokenizer, True)
            results["dev, gold evidence"] = report(
                f"{variant} | dev gold evidence", evaluate(model, gold_set, tokenizer, device)
            )
            results["dev, retrieved evidence"] = report(
                f"{variant} | dev dense top-{TOP_K}",
                evaluate(model, retrieved_set, tokenizer, device),
            )
        else:
            results["dev, claim only"] = report(
                f"{variant} | dev",
                evaluate(model, make_dataset(dev_gold, tokenizer, False), tokenizer, device),
            )

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
    (PROJECT_ROOT / "experiments" / f"verification_seed{seed}.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
