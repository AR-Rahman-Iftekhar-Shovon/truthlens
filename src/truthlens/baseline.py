from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_tfidf_logreg(seed: int) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            ),
            (
                "clf",
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
            ),
        ]
    )
