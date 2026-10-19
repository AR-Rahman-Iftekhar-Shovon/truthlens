import sys

from truthlens.baseline import build_tfidf_classifier
from truthlens.config import load_config
from truthlens.liar import load_processed_split
from truthlens.seed import set_seed

BINARY_C = 0.3


def main() -> None:
    seed = load_config()["seed"]
    set_seed(seed)

    train = load_processed_split("train")
    model = build_tfidf_classifier("logreg", BINARY_C, seed)
    model.fit(train["statement"], train["binary_label"])

    if len(sys.argv) > 1:
        statements = sys.argv[1:]
        actual = [""] * len(statements)
    else:
        sample = load_processed_split("validation").sample(8, random_state=seed)
        statements = sample["statement"].tolist()
        actual = sample["binary_label"].tolist()

    probabilities = model.predict_proba(statements)
    for statement, truth, probs in zip(statements, actual, probabilities):
        predicted = model.classes_[probs.argmax()]
        suffix = f"  [actual: {truth}]" if truth else ""
        print(f"{predicted:<5} {probs.max():.2f}  {statement}{suffix}")


if __name__ == "__main__":
    main()
