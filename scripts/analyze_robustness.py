import json
from datetime import date

import numpy as np
from sklearn.metrics import f1_score

from truthlens.calibration import fit_temperature, softmax_with_temperature
from truthlens.config import PROJECT_ROOT, load_config
from truthlens.decision import coverage_and_accuracy, load_rows

CONFIDENCE_THRESHOLD = 0.75


def main() -> None:
    folder = PROJECT_ROOT / load_config()["paths"]["data_processed"] / "averitec"
    probs_data = json.loads((folder / "verifier_probs_seed42.json").read_text(encoding="utf-8"))
    conditions = json.loads(
        (folder / "verifier_robustness_seed42.json").read_text(encoding="utf-8")
    )

    val_p, val_y, _ = load_rows(probs_data["val_retrieved"])
    temperature = fit_temperature(val_p, val_y)
    print(f"temperature: {temperature:.2f}, abstain when confidence < {CONFIDENCE_THRESHOLD}")
    print(
        f"{'condition':<22}{'acc':>7}{'macro_f1':>10}{'mean_conf':>11}{'coverage':>10}{'acc_answered':>14}"
    )

    record = {}
    for name, rows in conditions.items():
        probs, labels, _ = load_rows(rows)
        calibrated = softmax_with_temperature(probs, temperature)
        predicted = calibrated.argmax(axis=1)
        accuracy = float((predicted == labels).mean())
        macro_f1 = f1_score(labels, predicted, labels=[0, 1, 2], average="macro", zero_division=0)
        coverage, answered_accuracy = coverage_and_accuracy(
            calibrated, labels, np.zeros(len(labels)), CONFIDENCE_THRESHOLD, 0.0
        )
        mean_confidence = float(calibrated.max(axis=1).mean())
        print(
            f"{name:<22}{accuracy:>7.3f}{macro_f1:>10.3f}{mean_confidence:>11.3f}"
            f"{coverage:>10.3f}{answered_accuracy:>14.3f}"
        )
        record[name] = {
            "accuracy": round(accuracy, 4),
            "macro_f1": round(float(macro_f1), 4),
            "mean_confidence": round(mean_confidence, 4),
            "coverage": round(coverage, 3),
            "accuracy_on_answered": round(answered_accuracy, 3),
        }

    result = {"experiment": "robustness", "date": date.today().isoformat(), "conditions": record}
    (PROJECT_ROOT / "experiments").mkdir(exist_ok=True)
    (PROJECT_ROOT / "experiments" / "robustness.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
