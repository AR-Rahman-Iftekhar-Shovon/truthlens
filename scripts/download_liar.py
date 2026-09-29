import hashlib
import urllib.request
import zipfile
from pathlib import Path

from truthlens.config import PROJECT_ROOT, load_config

LIAR_URL = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    config = load_config()
    raw_dir = PROJECT_ROOT / config["paths"]["data_raw"] / "liar"
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / "liar_dataset.zip"
    if not zip_path.exists():
        print(f"Downloading {LIAR_URL}")
        urllib.request.urlretrieve(LIAR_URL, zip_path)

    print(f"sha256: {sha256_of(zip_path)}")

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(raw_dir)

    for path in sorted(raw_dir.iterdir()):
        print(f"{path.name}  {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
# sha256: 611c1addad919743dde15822b87a60bfb760d8f85597f25289e34621800654c7