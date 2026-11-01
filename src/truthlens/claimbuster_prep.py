import random

import pandas as pd


def add_check_worthy_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label"] = df["Verdict"].map(lambda v: "check-worthy" if v == 1 else "not-check-worthy")
    return df[["Sentence_id", "Text", "File_id", "Verdict", "label"]]


def split_by_debate(
    df: pd.DataFrame, seed: int, val_frac: float = 0.15, test_frac: float = 0.15
) -> dict[str, pd.DataFrame]:
    debates = sorted(df["File_id"].unique())
    random.Random(seed).shuffle(debates)

    n_test = max(1, round(len(debates) * test_frac))
    n_val = max(1, round(len(debates) * val_frac))
    test_ids = set(debates[:n_test])
    val_ids = set(debates[n_test : n_test + n_val])

    is_test = df["File_id"].isin(test_ids)
    is_val = df["File_id"].isin(val_ids)
    return {
        "train": df[~is_test & ~is_val],
        "validation": df[is_val],
        "test": df[is_test],
    }


def prepare_claimbuster(
    crowd: pd.DataFrame, gold: pd.DataFrame, seed: int
) -> tuple[dict[str, pd.DataFrame], dict]:
    crowd = add_check_worthy_label(crowd)
    gold = add_check_worthy_label(gold)

    report = {
        "crowd_rows": len(crowd),
        "crowd_duplicate_texts": int(crowd["Text"].duplicated().sum()),
    }
    label_counts = crowd.groupby("Text")["label"].nunique()
    report["duplicate_texts_with_conflicting_labels"] = int((label_counts > 1).sum())

    crowd = crowd.drop_duplicates("Text", keep="first")
    in_gold = crowd["Text"].isin(set(gold["Text"]))
    report["crowd_rows_also_in_gold"] = int(in_gold.sum())
    crowd = crowd[~in_gold]
    gold = gold.drop_duplicates("Text", keep="first")

    splits = split_by_debate(crowd, seed)
    train_debates = set(splits["train"]["File_id"])
    report["gold_debates"] = int(gold["File_id"].nunique())
    report["gold_debates_also_in_train"] = len(set(gold["File_id"]) & train_debates)

    splits["gold"] = gold
    return splits, report
