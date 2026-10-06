from truthlens.config import PROJECT_ROOT, load_config
from truthlens.liar import SPLIT_FILES, load_split
from truthlens.liar_prep import prepare_splits


def main() -> None:
    splits = {name: load_split(name) for name in SPLIT_FILES}
    prepared, report = prepare_splits(splits)

    out_dir = PROJECT_ROOT / load_config()["paths"]["data_processed"] / "liar"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("report:")
    for key, value in report.items():
        print(f"  {key}: {value}")

    for name, df in prepared.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"\n{name}: {len(df)} rows")
        print(df["binary_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
