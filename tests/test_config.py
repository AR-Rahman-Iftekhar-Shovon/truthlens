from truthlens.config import load_config


def test_base_config_has_seed_and_paths():
    config = load_config("base")

    assert isinstance(config["seed"], int)
    assert "data_raw" in config["paths"]