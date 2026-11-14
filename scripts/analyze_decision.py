import json
from datetime import date

import numpy as np
from sklearn.metrics import f1_score

from truthlens.calibration import (
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
    reliability_table,
    softmax_with_temperature,
)
from truthlens.config import PROJECT_ROOT, load_config
from truthlens.decision import INSUFFICIENT, decide
from truthlens.verification_data import LABELS


def load_rows(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs = np.array([row["probs"] for row in rows])
    labels = np.array([LABELS.index(row["label"]) for row in rows])
    scores = np.array([row.get("top1_score", 0.0) for row in rows])
    return probs, labels, scores


def evaluate(probs, labels, scores, prob_threshold: float, score_threshold: float) -> dict:
    decisions = decide(probs, scores, prob_threshold, score_threshold)
    predicted = ["UNKNOWN" if d == INSUFFICIENT else d for d in decisions]
    truth = [LABELS[i] for i in labels]
    answered = [d != INSUFFICIENT for d in decisions]
    hits = [d == t for d, t, a in zip(decisions, truth, answered) if a]
    return {
        "macro_f1": round(
            f1_score(truth, predicted, labels=LABELS, average="macro", zero_division=0), 4
        ),
        "coverage": round(float(np.mean(answered)), 3),
        "accuracy_on_answered": round(float(np.mean(hits)), 3) if hits else None,
    }


def main() -> None:
    path = (
        PROJECT_ROOT
        / load_config()["paths"]["data_processed"]
        / "averitec"
        / "verifier_probs_seed42.json"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    val_p, val_y, val_s = load_rows(data["val_retrieved"])
    dev_p, dev_y, dev_s = load_rows(data["dev_retrieved"])
    print(f"val {len(val_y)}, dev {len(dev_y)}")

    temperature = fit_temperature(val_p, val_y)
    val_c = softmax_with_temperature(val_p, temperature)
    dev_c = softmax_with_temperature(dev_p, temperature)
    print(f"\ntemperature (fit on val): {temperature:.2f}")
    for name, probs in (("uncalibrated", dev_p), ("calibrated", dev_c)):
        accuracy = float((probs.argmax(axis=1) == dev_y).mean())
        print(
            f"{name:<13} ECE={expected_calibration_error(probs, dev_y):.4f} "
            f"NLL={negative_log_likelihood(probs, dev_y):.4f} acc={accuracy:.3f}"
        )
        print(
            f"  reliability (low, high, n, mean_conf, accuracy): {reliability_table(probs, dev_y)}"
        )

    order = np.argsort(-dev_c.max(axis=1), kind="stable")
    correct = (dev_c.argmax(axis=1) == dev_y)[order]
    print("\nrisk-coverage on dev (calibrated confidence):")
    for coverage in (1.0, 0.8, 0.6, 0.4, 0.2):
        n = int(round(coverage * len(order)))
        print(f"  coverage {coverage:.1f}: accuracy {correct[:n].mean():.3f} (n={n})")

    best = (-1.0, 0.0, -1.0)
    for prob_threshold in np.arange(0.4, 0.951, 0.05):
        for score_threshold in np.arange(0.0, 0.751, 0.05):
            result = evaluate(val_c, val_y, val_s, prob_threshold, score_threshold)
            if result["macro_f1"] > best[0]:
                best = (result["macro_f1"], float(prob_threshold), float(score_threshold))
    _, prob_threshold, score_threshold = best
    print(
        f"\nthresholds chosen on val: confidence >= {prob_threshold:.2f}, retrieval top1 >= {score_threshold:.2f}"
    )

    results = {
        "argmax (abstain only if UNKNOWN)": evaluate(dev_c, dev_y, dev_s, 0.0, -1.0),
        "decision layer (thresholds from val)": evaluate(
            dev_c, dev_y, dev_s, prob_threshold, score_threshold
        ),
    }
    print("\ndev results (abstention counted as UNKNOWN for macro-F1):")
    for name, result in results.items():
        print(f"  {name:<38} {result}")

    record = {
        "experiment": "decision_calibration",
        "date": date.today().isoformat(),
        "temperature": temperature,
        "thresholds": {"confidence": prob_threshold, "top1_score": score_threshold},
        "dev": results,
    }
    (PROJECT_ROOT / "experiments").mkdir(exist_ok=True)
    (PROJECT_ROOT / "experiments" / "decision_calibration.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
