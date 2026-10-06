import json

from truthlens.config import PROJECT_ROOT, load_config
from truthlens.hashing import sha256_of
from truthlens.liar import SPLIT_FILES, load_split
from truthlens.liar_prep import prepare_splits


def main() -> None:
    splits = {name: load_split(name) for name in SPLIT_FILES}
    prepared, report = prepare_splits(splits)

    out_dir = PROJECT_ROOT / load_config()["paths"]["data_processed"] / "liar"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"source": "liar_dataset.zip", "report": report, "splits": {}}
    for name, df in prepared.items():
        csv_path = out_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False, lineterminator="\n")
        manifest["splits"][name] = {
            "rows": len(df),
            "label_counts": df["label"].value_counts().to_dict(),
            "binary_label_counts": df["binary_label"].value_counts().to_dict(),
            "sha256": sha256_of(csv_path),
        }
        print(f"{name}: {len(df)} rows, sha256 {manifest['splits'][name]['sha256'][:12]}...")

    manifest_path = PROJECT_ROOT / "docs" / "liar_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"manifest written to {manifest_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
