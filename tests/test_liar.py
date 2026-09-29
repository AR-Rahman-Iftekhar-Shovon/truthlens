import pytest

from truthlens.liar import COLUMNS, SPLIT_FILES, liar_dir, load_split

pytestmark = pytest.mark.skipif(not liar_dir().exists(), reason="LIAR data not downloaded")


def test_split_sizes_match_published_numbers():
    sizes = {name: len(load_split(name)) for name in SPLIT_FILES}
    assert sizes == {"train": 10269, "validation": 1284, "test": 1283}


def test_columns_and_label_set():
    train = load_split("train")
    assert list(train.columns) == COLUMNS
    assert set(train["label"]) == {
        "pants-fire",
        "false",
        "barely-true",
        "half-true",
        "mostly-true",
        "true",
    }