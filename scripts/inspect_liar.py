from truthlens.liar import SPLIT_FILES, load_split


def main() -> None:
    splits = {name: load_split(name) for name in SPLIT_FILES}

    for name, df in splits.items():
        print(f"\n== {name}: {len(df)} rows")
        print(df["label"].value_counts().to_string())

        word_counts = df["statement"].str.split().str.len()
        print("\nstatement length (words):")
        print(word_counts.describe().round(1).to_string())

        missing = df.isna().sum()
        print("\nmissing values:", missing[missing > 0].to_dict())
        print("duplicate statements inside split:", int(df["statement"].duplicated().sum()))

    train_statements = set(splits["train"]["statement"])
    for other in ("validation", "test"):
        overlap = len(train_statements & set(splits[other]["statement"]))
        print(f"\nstatements shared between train and {other}: {overlap}")


if __name__ == "__main__":
    main()