import pandas as pd

FALSE_LABELS = {"barely-true", "false", "pants-fire"}
KEPT_COLUMNS = ["id", "statement", "label", "binary_label"]


def normalize_statement(text: str) -> str:
    return " ".join(text.lower().split())


def add_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["binary_label"] = df["label"].map(lambda label: "false" if label in FALSE_LABELS else "true")
    return df


def prepare_splits(splits: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict]:
    prepared = {}
    for name, df in splits.items():
        df = add_binary_label(df)
        df["statement_key"] = df["statement"].map(normalize_statement)
        prepared[name] = df[KEPT_COLUMNS + ["statement_key"]]

    train = prepared["train"]
    report = {"train_rows_before": len(train)}
    report["train_in_split_duplicates"] = int(train["statement_key"].duplicated().sum())

    label_counts = train.groupby("statement_key")["label"].nunique()
    report["train_duplicate_groups_with_conflicting_labels"] = int((label_counts > 1).sum())

    train = train.drop_duplicates("statement_key", keep="first")
    eval_keys = set(prepared["validation"]["statement_key"]) | set(
        prepared["test"]["statement_key"]
    )
    overlapping = train["statement_key"].isin(eval_keys)
    report["train_rows_overlapping_eval"] = int(overlapping.sum())
    prepared["train"] = train[~overlapping]

    val_keys = set(prepared["validation"]["statement_key"])
    test_keys = set(prepared["test"]["statement_key"])
    report["validation_test_overlap"] = len(val_keys & test_keys)
    report["validation_in_split_duplicates"] = int(
        prepared["validation"]["statement_key"].duplicated().sum()
    )
    report["test_in_split_duplicates"] = int(prepared["test"]["statement_key"].duplicated().sum())

    cleaned = {name: df[KEPT_COLUMNS] for name, df in prepared.items()}
    return cleaned, report
