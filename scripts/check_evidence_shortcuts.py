import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from truthlens.averitec import load_averitec_tables


def claim_features(claims: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    evidence = evidence.assign(answer_length=evidence["answer"].str.len())
    types = pd.crosstab(evidence["claim_id"], evidence["answer_type"])
    lengths = evidence.groupby("claim_id")["answer_length"].mean().rename("mean_answer_length")
    base = claims.set_index("claim_id")[["split", "label_mapped", "n_evidence"]]
    return base.join(types).join(lengths).fillna(0)


def main() -> None:
    claims, evidence = load_averitec_tables()
    features = claim_features(claims, evidence)
    columns = [c for c in features.columns if c not in ("split", "label_mapped")]

    print(features.groupby("label_mapped")[columns].mean().round(2).to_string())

    train = features[features["split"] == "train"]
    dev = features[features["split"] == "dev"]

    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
    )
    model.fit(train[columns], train["label_mapped"])
    majority = DummyClassifier(strategy="most_frequent").fit(train[columns], train["label_mapped"])

    for name, predictor in (("structure features", model), ("majority class", majority)):
        score = f1_score(dev["label_mapped"], predictor.predict(dev[columns]), average="macro")
        print(f"dev macro-F1, {name}: {score:.3f}")


if __name__ == "__main__":
    main()
