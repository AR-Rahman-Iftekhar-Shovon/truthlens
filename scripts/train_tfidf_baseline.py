import argparse
import json
from datetime import date

from sklearn.dummy import DummyClassifier

from truthlens.baseline import build_tfidf_logreg
from truthlens.config import PROJECT_ROOT, load_config
from truthlens.evaluation import summarize
from truthlens.liar import load_processed_split
from truthlens.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["binary_label", "label"], default="binary_label")
    args = parser.parse_args()

    seed = load_config()["seed"]
    set_seed(seed)

    train = load_processed_split("train")
    validation = load_processed_split("validation")
    labels = sorted(train[args.target].unique())

    model = build_tfidf_logreg(seed)
    model.fit(train["statement"], train[args.target])
    predictions = model.predict(validation["statement"])
    summary = summarize(validation[args.target], predictions, labels)

    majority = DummyClassifier(strategy="most_frequent")
    majority.fit(train["statement"], train[args.target])
    majority_summary = summarize(
        validation[args.target], majority.predict(validation["statement"]), labels
    )

    experiment = f"tfidf_logreg_{args.target}"
    result = {
        "experiment": experiment,
        "date": date.today().isoformat(),
        "dataset": "LIAR processed, see docs/liar_manifest.json",
        "evaluated_on": "validation",
        "seed": seed,
        "model": {
            "vectorizer": {"ngram_range": [1, 2], "min_df": 2, "sublinear_tf": True},
            "classifier": {
                "name": "LogisticRegression",
                "class_weight": "balanced",
                "max_iter": 1000,
            },
        },
        "majority_baseline_macro_f1": majority_summary["macro_f1"],
        **summary,
    }

    out_dir = PROJECT_ROOT / "experiments"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{experiment}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    print(f"{experiment} (validation)")
    print(f"  accuracy: {summary['accuracy']}")
    print(
        f"  macro_f1: {summary['macro_f1']}   (majority-class macro_f1: {majority_summary['macro_f1']})"
    )
    print(f"  labels: {labels}")
    for row in summary["confusion_matrix"]:
        print(f"  {row}")


if __name__ == "__main__":
    main()
