import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from truthlens.baseline import build_tfidf_classifier
from truthlens.config import PROJECT_ROOT, load_config
from truthlens.evaluation import summarize
from truthlens.liar import load_processed_split
from truthlens.transformer_data import StatementDataset
from truthlens.transformer_train import predict

LABELS = ["false", "true"]
MODEL_NAME = "distilroberta-base"
BASELINE_C = 0.3
MAX_LENGTH = 64


def evaluate_baseline(train, test, seed: int) -> dict:
    model = build_tfidf_classifier("logreg", BASELINE_C, seed)
    model.fit(train["statement"], train["binary_label"])
    return summarize(test["binary_label"], model.predict(test["statement"]), LABELS)


def evaluate_transformer(seed: int, test, tokenizer, device) -> dict:
    weights = PROJECT_ROOT / "outputs" / f"distilroberta_base_binary_seed{seed}.pt"
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.load_state_dict(torch.load(weights))
    model.to(device)

    dataset = StatementDataset(
        test["statement"].tolist(),
        [LABELS.index(label) for label in test["binary_label"]],
        tokenizer,
        MAX_LENGTH,
    )
    loader = DataLoader(dataset, batch_size=64, collate_fn=DataCollatorWithPadding(tokenizer))
    predicted = [LABELS[i] for i in predict(model, loader, device)]
    return summarize(test["binary_label"], predicted, LABELS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()

    seed = load_config()["seed"]
    train = load_processed_split("train")
    test = load_processed_split("test")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device = torch.device("cuda")

    results = {"baseline_tfidf_logreg": evaluate_baseline(train, test, seed)}
    for s in args.seeds:
        results[f"distilroberta_seed{s}"] = evaluate_transformer(s, test, tokenizer, device)

    print(f"test set: {len(test)} rows")
    for name, summary in results.items():
        print(
            f"{name:<24} accuracy={summary['accuracy']}  macro_f1={summary['macro_f1']}  "
            f"confusion={summary['confusion_matrix']}"
        )

    f1_scores = [results[f"distilroberta_seed{s}"]["macro_f1"] for s in args.seeds]
    print(
        f"distilroberta macro_f1 mean={np.mean(f1_scores):.4f} std={np.std(f1_scores, ddof=1):.4f}"
    )


if __name__ == "__main__":
    main()
