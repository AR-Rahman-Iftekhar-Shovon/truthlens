import urllib.request

from truthlens.config import PROJECT_ROOT, load_config
from truthlens.hashing import sha256_of

BASE_URL = "https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data/{name}"
FILES = ["train.json", "dev.json"]


def main() -> None:
    raw_dir = PROJECT_ROOT / load_config()["paths"]["data_raw"] / "averitec"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for name in FILES:
        path = raw_dir / name
        if not path.exists():
            urllib.request.urlretrieve(BASE_URL.format(name=name), path)
        print(f"{name}: {path.stat().st_size} bytes, sha256 {sha256_of(path)}")


if __name__ == "__main__":
    main()
