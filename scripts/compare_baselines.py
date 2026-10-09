import argparse
import json
from datetime import date

from truthlens.baseline import build_tfidf_classifier
from truthlens.config import PROJECT_ROOT, load_config
from truthlens.evaluation import summarize
from truthlens.liar import load_processed_split
from truthlens.seed import set_seed

KINDS = ["logreg", "linear_svm"]
C_VALUES = [0.1, 0.3, 1.0, 3.0, 10.0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["binary_label", "label"], default="binary_label")
    args = parser.parse_args()

    seed = load_config()["seed"]
    set_seed(seed)

    train = load_processed_split("train")
    validation = load_processed_split("validation")
    labels = sorted(train[args.target].unique())

    rows = []
    for kind in KINDS:
        for c in C_VALUES:
            model = build_tfidf_classifier(kind, c, seed)
            model.fit(train["statement"], train[args.target])
            predictions = model.predict(validation["statement"])
            summary = summarize(validation[args.target], predictions, labels)
            rows.append(
                {
                    "model": kind,
                    "C": c,
                    "accuracy": summary["accuracy"],
                    "macro_f1": summary["macro_f1"],
                }
            )
            print(
                f"{kind:<11} C={c:<5} accuracy={summary['accuracy']}  macro_f1={summary['macro_f1']}"
            )

    best = max(rows, key=lambda row: row["macro_f1"])
    print(f"\nbest on validation (macro_f1): {best}")

    experiment = f"baseline_grid_{args.target}"
    result = {
        "experiment": experiment,
        "date": date.today().isoformat(),
        "dataset": "LIAR processed, see docs/liar_manifest.json",
        "evaluated_on": "validation",
        "seed": seed,
        "vectorizer": {"ngram_range": [1, 2], "min_df": 2, "sublinear_tf": True},
        "class_weight": "balanced",
        "results": rows,
        "best_on_validation": best,
    }
    out_dir = PROJECT_ROOT / "experiments"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{experiment}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
