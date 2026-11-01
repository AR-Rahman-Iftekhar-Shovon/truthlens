import pandas as pd

from truthlens.config import PROJECT_ROOT, load_config


def load_claimbuster_split(split: str) -> pd.DataFrame:
    path = PROJECT_ROOT / load_config()["paths"]["data_processed"] / "claimbuster" / f"{split}.csv"
    return pd.read_csv(path)
