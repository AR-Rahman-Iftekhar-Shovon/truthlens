import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from truthlens.baseline import build_tfidf_classifier
from truthlens.claimbuster import load_claimbuster_split
from truthlens.config import PROJECT_ROOT, load_config
from truthlens.evaluation import summarize
from truthlens.transformer_data import StatementDataset
from truthlens.transformer_train import predict

LABELS = ["not-check-worthy", "check-worthy"]
POSITIVE = "check-worthy"
MODEL_NAME = "distilroberta-base"
BASELINE_C = 3.0
MAX_LENGTH = 128
SEEDS = [42, 43, 44]


def transformer_predictions(seed: int, eval_sets: dict, tokenizer, device) -> dict:
    weights = PROJECT_ROOT / "outputs" / f"distilroberta_base_claimbuster_seed{seed}.pt"
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.load_state_dict(torch.load(weights))
    model.to(device)

    collator = DataCollatorWithPadding(tokenizer)
    predictions = {}
    for name, df in eval_sets.items():
        texts = df["Text"].tolist()
        dataset = StatementDataset(texts, [0] * len(texts), tokenizer, MAX_LENGTH)
        loader = DataLoader(dataset, batch_size=64, collate_fn=collator)
        predictions[name] = [LABELS[i] for i in predict(model, loader, device)]
    return predictions


def main() -> None:
    seed0 = load_config()["seed"]
    train = load_claimbuster_split("train")
    test = load_claimbuster_split("test")
    gold = load_claimbuster_split("gold")
    unseen = gold[~gold["File_id"].isin(set(train["File_id"]))]
    eval_sets = {"test": test, "gold_all": gold, "gold_unseen_debates": unseen}

    baseline = build_tfidf_classifier("logreg", BASELINE_C, seed0)
    baseline.fit(train["Text"], train["label"])
    predictions = {"baseline": {n: baseline.predict(df["Text"]) for n, df in eval_sets.items()}}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device = torch.device("cuda")
    for seed in SEEDS:
        predictions[f"distilroberta_seed{seed}"] = transformer_predictions(
            seed, eval_sets, tokenizer, device
        )

    print(f"{'model':<20}{'set':<21}{'rows':>6}{'prec':>7}{'recall':>8}{'f1':>7}{'macro_f1':>10}")
    metrics = {}
    for model_name, by_set in predictions.items():
        for set_name, df in eval_sets.items():
            summary = summarize(df["label"], by_set[set_name], LABELS)
            positive = summary["per_class"][POSITIVE]
            f1 = positive["f1-score"]
            metrics[(model_name, set_name)] = (f1, summary["macro_f1"])
            print(
                f"{model_name:<20}{set_name:<21}{len(df):>6}{positive['precision']:>7.3f}"
                f"{positive['recall']:>8.3f}{f1:>7.3f}{summary['macro_f1']:>10.3f}"
            )

    print(f"\ngold_unseen_debates: {unseen['File_id'].nunique()} debates")
    for set_name in eval_sets:
        runs = [metrics[(f"distilroberta_seed{s}", set_name)] for s in SEEDS]
        f1s, macros = zip(*runs)
        print(
            f"distilroberta mean on {set_name}: f1={np.mean(f1s):.3f} (std {np.std(f1s, ddof=1):.3f}), "
            f"macro_f1={np.mean(macros):.3f} (std {np.std(macros, ddof=1):.3f})"
        )


if __name__ == "__main__":
    main()
