import pandas as pd

from truthlens.robustness import add_typos, build_conditions, make_sensational, strip_noise


def test_perturbations():
    original = "government spending increased"
    typo = add_typos(original, 1.0, 42)

    assert make_sensational("claim") == "BREAKING!!! SHOCKING: CLAIM!!!"
    assert strip_noise("Hello, World!") == "hello world"
    assert typo != original
    assert sorted(typo.replace(" ", "")) == sorted(original.replace(" ", ""))


def test_build_conditions_swaps_only_the_evidence():
    dev = pd.DataFrame(
        {
            "claim_id": ["d1", "d2"],
            "claim": ["a b", "c d"],
            "label_mapped": ["SUPPORTED", "REFUTED"],
        }
    )
    retrieved = {"d1": ["r1", "r2", "r3"], "d2": ["r4", "r5", "r6"]}
    misleading = {"d1": ["m1", "m2"]}
    evidence = pd.DataFrame({"claim_id": ["t1"] * 6, "text": [f"x{i}" for i in range(6)]})

    conditions = build_conditions(dev, retrieved, misleading, evidence, seed=42)

    assert len(conditions) == 8
    for examples in conditions.values():
        assert examples["claim_id"].tolist() == ["d1", "d2"]
    assert conditions["misleading_evidence"]["evidence"].tolist() == ["m1 m2", "r4 r5 r6"]
    assert conditions["mixed_evidence"]["evidence"].tolist()[0] == "r1 r2 r3 m1 m2"
    assert conditions["no_evidence"]["evidence"].tolist() == ["", ""]
