import json
from datetime import date

import numpy as np

from truthlens.calibration import fit_temperature, softmax_with_temperature
from truthlens.config import PROJECT_ROOT, load_config
from truthlens.decision import coverage_and_accuracy, load_rows, select_operating_point

TARGETS = (0.75, 0.80, 0.85)


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

    temperature = fit_temperature(val_p, val_y)
    val_c = softmax_with_temperature(val_p, temperature)
    dev_c = softmax_with_temperature(dev_p, temperature)

    prob_grid = np.arange(0.4, 0.951, 0.05)
    score_grid = np.arange(0.0, 0.751, 0.05)
    print(
        f"{'target':<8}{'conf>=':>8}{'top1>=':>8}"
        f"{'val cov':>9}{'val acc':>9}{'dev cov':>9}{'dev acc':>9}"
    )

    record = {}
    for target in TARGETS:
        chosen = select_operating_point(val_c, val_y, val_s, target, prob_grid, score_grid)
        if chosen is None:
            print(f"{target:<8}no setting reaches the target on val")
            continue
        _, prob_threshold, score_threshold = chosen
        val_cov, val_acc = coverage_and_accuracy(
            val_c, val_y, val_s, prob_threshold, score_threshold
        )
        dev_cov, dev_acc = coverage_and_accuracy(
            dev_c, dev_y, dev_s, prob_threshold, score_threshold
        )
        print(
            f"{target:<8}{prob_threshold:>8.2f}{score_threshold:>8.2f}{val_cov:>9.3f}"
            f"{val_acc:>9.3f}{dev_cov:>9.3f}{dev_acc:>9.3f}"
        )
        record[str(target)] = {
            "confidence": round(prob_threshold, 2),
            "top1_score": round(score_threshold, 2),
            "val": {"coverage": round(val_cov, 3), "accuracy": round(val_acc, 3)},
            "dev": {"coverage": round(dev_cov, 3), "accuracy": round(dev_acc, 3)},
        }

    result = {"experiment": "operating_points", "date": date.today().isoformat(), "points": record}
    (PROJECT_ROOT / "experiments").mkdir(exist_ok=True)
    (PROJECT_ROOT / "experiments" / "operating_points.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
