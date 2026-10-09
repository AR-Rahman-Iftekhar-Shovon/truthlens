from truthlens.baseline import build_tfidf_classifier
from truthlens.config import load_config
from truthlens.features import top_weighted_features
from truthlens.liar import load_processed_split
from truthlens.seed import set_seed

BINARY_C = 0.3
TOP_N = 25


def main() -> None:
    seed = load_config()["seed"]
    set_seed(seed)

    train = load_processed_split("train")
    model = build_tfidf_classifier("logreg", BINARY_C, seed)
    model.fit(train["statement"], train["binary_label"])

    true_features, false_features = top_weighted_features(model, TOP_N)
    print(f"classes: {list(model.classes_)}  (C={BINARY_C}, trained on train only)")
    print("\nfeatures pushing towards 'true':")
    print(", ".join(true_features))
    print("\nfeatures pushing towards 'false':")
    print(", ".join(false_features))


if __name__ == "__main__":
    main()