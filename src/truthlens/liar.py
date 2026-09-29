import csv
from pathlib import Path

import pandas as pd

from truthlens.config import PROJECT_ROOT, load_config

COLUMNS = [
    "id",
    "label",
    "statement",
    "subject",
    "speaker",
    "job_title",
    "state",
    "party",
    "barely_true_count",
    "false_count",
    "half_true_count",
    "mostly_true_count",
    "pants_fire_count",
    "context",
]

SPLIT_FILES = {"train": "train.tsv", "validation": "valid.tsv", "test": "test.tsv"}


def liar_dir() -> Path:
    return PROJECT_ROOT / load_config()["paths"]["data_raw"] / "liar"


def load_split(split: str) -> pd.DataFrame:
    return pd.read_csv(
        liar_dir() / SPLIT_FILES[split],
        sep="\t",
        header=None,
        names=COLUMNS,
        quoting=csv.QUOTE_NONE,
    )