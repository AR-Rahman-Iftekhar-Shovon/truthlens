import numpy as np
from sklearn.pipeline import Pipeline


def top_weighted_features(model: Pipeline, n: int) -> tuple[list[str], list[str]]:
    coefficients = model.named_steps["clf"].coef_
    if coefficients.shape[0] != 1:
        raise ValueError("expected a binary classifier")

    names = np.array(model.named_steps["tfidf"].get_feature_names_out())
    order = np.argsort(coefficients[0])
    highest = names[order[::-1][:n]].tolist()
    lowest = names[order[:n]].tolist()
    return highest, lowest