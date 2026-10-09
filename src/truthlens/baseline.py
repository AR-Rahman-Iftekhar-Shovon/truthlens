from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_tfidf_classifier(kind: str, c: float, seed: int) -> Pipeline:
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)

    if kind == "logreg":
        classifier = LogisticRegression(
            C=c, max_iter=1000, class_weight="balanced", random_state=seed
        )
    elif kind == "linear_svm":
        classifier = LinearSVC(C=c, max_iter=5000, class_weight="balanced", random_state=seed)
    else:
        raise ValueError(f"unknown classifier kind: {kind}")

    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def build_tfidf_logreg(seed: int) -> Pipeline:
    return build_tfidf_classifier("logreg", 1.0, seed)
