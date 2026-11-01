import pandas as pd

from truthlens.claimbuster_prep import prepare_claimbuster
from truthlens.config import PROJECT_ROOT, load_config


def main() -> None:
    config = load_config()
    raw_dir = PROJECT_ROOT / config["paths"]["data_raw"] / "claimbuster"
    crowd = pd.read_csv(raw_dir / "crowdsourced.csv")
    gold = pd.read_csv(raw_dir / "groundtruth.csv")

    splits, report = prepare_claimbuster(crowd, gold, config["seed"])

    out_dir = PROJECT_ROOT / config["paths"]["data_processed"] / "claimbuster"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("report:")
    for key, value in report.items():
        print(f"  {key}: {value}")

    print(f"\n{'split':<11}{'rows':>7}{'debates':>9}{'check-worthy %':>16}")
    for name, df in splits.items():
        df.to_csv(out_dir / f"{name}.csv", index=False, lineterminator="\n")
        share = 100 * (df["label"] == "check-worthy").mean()
        print(f"{name:<11}{len(df):>7}{df['File_id'].nunique():>9}{share:>16.1f}")


if __name__ == "__main__":
    main()
