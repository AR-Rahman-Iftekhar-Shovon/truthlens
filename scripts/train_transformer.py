import argparse
import json
from datetime import date

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from truthlens.claimbuster import load_claimbuster_split
from truthlens.config import PROJECT_ROOT, load_config
from truthlens.liar import load_processed_split
from truthlens.transformer_data import StatementDataset
from truthlens.transformer_train import fine_tune, set_torch_seed

DATASETS = {
    "liar": {
        "load": load_processed_split,
        "text": "statement",
        "label": "binary_label",
        "labels": ["false", "true"],
        "max_length": 64,
        "tag": "binary",
    },
    "claimbuster": {
        "load": load_claimbuster_split,
        "text": "Text",
        "label": "label",
        "labels": ["not-check-worthy", "check-worthy"],
        "max_length": 128,
        "tag": "claimbuster",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS), default="liar")
    parser.add_argument("--model", default="distilroberta-base")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    spec = DATASETS[args.dataset]
    seed = args.seed if args.seed is not None else load_config()["seed"]
    set_torch_seed(seed)

    label_to_id = {label: i for i, label in enumerate(spec["labels"])}
    id_to_label = {i: label for label, i in label_to_id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    datasets = {}
    for split in ("train", "validation"):
        df = spec["load"](split)
        datasets[split] = StatementDataset(
            df[spec["text"]].tolist(),
            [label_to_id[label] for label in df[spec["label"]]],
            tokenizer,
            spec["max_length"],
        )
    print({split: len(ds) for split, ds in datasets.items()})

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    history, best_summary, best_state = fine_tune(
        model,
        datasets["train"],
        datasets["validation"],
        tokenizer,
        id_to_label,
        args.epochs,
        args.lr,
        args.batch_size,
        seed,
    )

    experiment = f"{args.model.replace('-', '_')}_{spec['tag']}_seed{seed}"
    result = {
        "experiment": experiment,
        "date": date.today().isoformat(),
        "dataset": args.dataset,
        "evaluated_on": "validation",
        "seed": seed,
        "model": args.model,
        "max_length": spec["max_length"],
        "hyperparameters": {"epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size},
        "history": history,
        "best_epoch_macro_f1": best_summary["macro_f1"],
        "best_epoch_accuracy": best_summary["accuracy"],
        "confusion_matrix": best_summary["confusion_matrix"],
    }

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    torch.save(best_state, out_dir / f"{experiment}.pt")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
