import hashlib
import urllib.request

from truthlens.config import PROJECT_ROOT, load_config

BASE_URL = "https://zenodo.org/records/3609356/files/{name}?download=1"
FILES = {
    "all_sentences.csv": "686ac1dd5123d9ca0d229ee9760d4962",
    "crowdsourced.csv": "af9649bc3cc93edbc804893720a50bde",
    "groundtruth.csv": "1577f6d45bf33eabe9cf760f0fb66da3",
}


def md5_of(path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    raw_dir = PROJECT_ROOT / load_config()["paths"]["data_raw"] / "claimbuster"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for name, expected_md5 in FILES.items():
        path = raw_dir / name
        if not path.exists():
            urllib.request.urlretrieve(BASE_URL.format(name=name), path)
        status = "OK" if md5_of(path) == expected_md5 else "MISMATCH"
        print(f"{name}: md5 {status}, {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
