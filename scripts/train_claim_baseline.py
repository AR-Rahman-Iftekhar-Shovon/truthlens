from sklearn.metrics import average_precision_score

from truthlens.baseline import build_tfidf_classifier
from truthlens.claimbuster import load_claimbuster_split
from truthlens.config import load_config
from truthlens.evaluation import summarize
from truthlens.seed import set_seed

LABELS = ["check-worthy", "not-check-worthy"]
POSITIVE = "check-worthy"
C_VALUES = [0.3, 1.0, 3.0]


def main() -> None:
    seed = load_config()["seed"]
    set_seed(seed)

    train = load_claimbuster_split("train")
    validation = load_claimbuster_split("validation")
    y_val = validation["label"]
    print(f"validation: {len(validation)} rows, positive rate {(y_val == POSITIVE).mean():.3f}")
    print(f"{'C':<6}{'precision':>10}{'recall':>8}{'f1':>8}{'macro_f1':>10}{'avg_prec':>10}")

    for c in C_VALUES:
        model = build_tfidf_classifier("logreg", c, seed)
        model.fit(train["Text"], train["label"])
        summary = summarize(y_val, model.predict(validation["Text"]), LABELS)
        positive = summary["per_class"][POSITIVE]

        positive_index = list(model.classes_).index(POSITIVE)
        scores = model.predict_proba(validation["Text"])[:, positive_index]
        average_precision = average_precision_score(y_val == POSITIVE, scores)

        print(
            f"{c:<6}{positive['precision']:>10.3f}{positive['recall']:>8.3f}"
            f"{positive['f1-score']:>8.3f}{summary['macro_f1']:>10.3f}{average_precision:>10.3f}"
        )


if __name__ == "__main__":
    main()
