import pandas as pd

from truthlens.claimbuster_prep import add_check_worthy_label, prepare_claimbuster, split_by_debate

COLUMNS = ["Sentence_id", "Text", "File_id", "Verdict"]


def test_split_by_debate_keeps_debates_separate():
    rows = [(i, f"sentence {i}", i % 10, 1 if i % 4 == 0 else -1) for i in range(100)]
    df = add_check_worthy_label(pd.DataFrame(rows, columns=COLUMNS))

    splits = split_by_debate(df, seed=42)

    debates = {name: set(part["File_id"]) for name, part in splits.items()}
    assert debates["train"].isdisjoint(debates["validation"])
    assert debates["train"].isdisjoint(debates["test"])
    assert debates["validation"].isdisjoint(debates["test"])
    assert sum(len(part) for part in splits.values()) == len(df)


def test_prepare_removes_duplicates_and_gold_texts():
    crowd_rows = [(i, f"sentence {i}", i, -1) for i in range(10)]
    crowd_rows[0] = (0, "dup", 0, 1)
    crowd_rows[1] = (1, "dup", 1, -1)
    crowd_rows[2] = (2, "gold text", 2, 0)
    crowd = pd.DataFrame(crowd_rows, columns=COLUMNS)
    gold = pd.DataFrame([(100, "gold text", 2, 1), (101, "other", 3, -1)], columns=COLUMNS)

    splits, report = prepare_claimbuster(crowd, gold, seed=42)

    assert report["crowd_duplicate_texts"] == 1
    assert report["duplicate_texts_with_conflicting_labels"] == 1
    assert report["crowd_rows_also_in_gold"] == 1
    training_texts = (
        set(splits["train"]["Text"])
        | set(splits["validation"]["Text"])
        | set(splits["test"]["Text"])
    )
    assert "gold text" not in training_texts
