import pandas as pd

from truthlens.liar_prep import add_binary_label, prepare_splits


def make_split(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["id", "statement", "label"])


def test_add_binary_label():
    df = add_binary_label(make_split([("1", "a", "pants-fire"), ("2", "b", "half-true")]))
    assert list(df["binary_label"]) == ["false", "true"]


def test_prepare_splits_removes_duplicates_and_eval_overlap():
    splits = {
        "train": make_split(
            [
                ("1", "Taxes went up.", "false"),
                ("2", "taxes  went up.", "false"),
                ("3", "Jobs grew.", "true"),
                ("4", "Shared claim", "half-true"),
            ]
        ),
        "validation": make_split([("5", "shared claim", "half-true")]),
        "test": make_split([("6", "Other claim", "true")]),
    }

    prepared, report = prepare_splits(splits)

    assert list(prepared["train"]["id"]) == ["1", "3"]
    assert len(prepared["validation"]) == 1
    assert report["train_in_split_duplicates"] == 1
    assert report["train_rows_overlapping_eval"] == 1
