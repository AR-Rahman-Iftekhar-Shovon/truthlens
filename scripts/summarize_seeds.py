import json
from datetime import date

import numpy as np
from sklearn.metrics import f1_score

from truthlens.calibration import fit_temperature, softmax_with_temperature
from truthlens.config import PROJECT_ROOT, load_config
from truthlens.decision import coverage_and_accuracy, load_rows

SEEDS = (42, 43, 44)
VARIANTS = ("bm25", "dense", "hybrid", "dense_rerank_zero_shot")
CONFIDENCE = 0.75


def read(folder, name):
    return json.loads((folder / name).read_text(encoding="utf-8"))


def mean_std(values: list[float]) -> str:
    return f"{np.mean(values):.3f}±{np.std(values, ddof=1):.3f}"


def main() -> None:
    folder = PROJECT_ROOT / load_config()["paths"]["data_processed"] / "averitec"
    metrics = {
        name: {"accuracy": [], "macro_f1": [], "coverage": [], "answered": []} for name in VARIANTS
    }

    for seed in SEEDS:
        probs_data = read(folder, f"verifier_probs_seed{seed}.json")
        analysis = read(folder, f"verifier_analysis_seed{seed}.json")
        val_p, val_y, _ = load_rows(probs_data["val_retrieved"])
        temperature = fit_temperature(val_p, val_y)
        for name in VARIANTS:
            probs, labels, _ = load_rows(analysis["variants"][name])
            predicted = probs.argmax(axis=1)
            metrics[name]["accuracy"].append(float((predicted == labels).mean()))
            metrics[name]["macro_f1"].append(
                float(
                    f1_score(labels, predicted, labels=[0, 1, 2], average="macro", zero_division=0)
                )
            )
            coverage, answered = coverage_and_accuracy(
                softmax_with_temperature(probs, temperature),
                labels,
                np.zeros(len(labels)),
                CONFIDENCE,
                0.0,
            )
            metrics[name]["coverage"].append(coverage)
            metrics[name]["answered"].append(answered)

    print(f"seeds {SEEDS}, confidence threshold {CONFIDENCE}")
    print(f"{'evidence':<26}{'accuracy':>14}{'macro_f1':>14}{'coverage':>14}{'acc_answered':>14}")
    for name in VARIANTS:
        values = metrics[name]
        print(
            f"{name:<26}{mean_std(values['accuracy']):>14}{mean_std(values['macro_f1']):>14}"
            f"{mean_std(values['coverage']):>14}{mean_std(values['answered']):>14}"
        )

    record = {
        "experiment": "seed_summary",
        "date": date.today().isoformat(),
        "seeds": list(SEEDS),
        "variants": {
            n: {k: [round(x, 4) for x in v] for k, v in m.items()} for n, m in metrics.items()
        },
    }
    (PROJECT_ROOT / "experiments" / "seed_summary.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
