from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(name: str = "base") -> dict:
    config_path = PROJECT_ROOT / "configs" / f"{name}.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)
