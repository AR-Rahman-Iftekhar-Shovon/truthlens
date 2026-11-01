import pandas as pd

from truthlens.config import PROJECT_ROOT, load_config


def main() -> None:
    raw_dir = PROJECT_ROOT / load_config()["paths"]["data_raw"] / "claimbuster"
    crowd = pd.read_csv(raw_dir / "crowdsourced.csv")
    truth = pd.read_csv(raw_dir / "groundtruth.csv")

    for name, df in (("crowdsourced", crowd), ("groundtruth", truth)):
        print(f"\n== {name}: {len(df)} rows")
        print(df["Verdict"].value_counts(dropna=False).sort_index().to_string())
        print("debates (File_id):", df["File_id"].nunique())
        print("duplicate texts:", int(df["Text"].duplicated().sum()))
        words = df["Text"].str.split().str.len()
        print(
            "words per sentence (median, 95%, max):",
            words.median(),
            words.quantile(0.95),
            words.max(),
        )

    print(
        "\nSentence_id shared between files:",
        len(set(crowd["Sentence_id"]) & set(truth["Sentence_id"])),
    )
    print("Texts shared between files:", len(set(crowd["Text"]) & set(truth["Text"])))
    print("\nsample rows from groundtruth:")
    print(truth[["Text", "Verdict"]].head(5).to_string())


if __name__ == "__main__":
    main()
