import json
from collections import Counter
from datetime import date

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from truthlens.averitec import gold_by_claim, load_averitec_tables
from truthlens.calibration import fit_temperature, softmax_with_temperature
from truthlens.config import PROJECT_ROOT, load_config
from truthlens.decision import coverage_and_accuracy, load_rows
from truthlens.error_analysis import CONFIDENCE_THRESHOLD, error_category
from truthlens.verification_data import LABELS

GATES = (0.45, 0.55)
CLAIM_ONLY = (0.6407, 0.4806)
GOLD_EVIDENCE = (0.7316, 0.5929)
EVIDENCE_CONDITIONS = ("clean", "misleading_evidence", "mixed_evidence", "random_evidence")


def read_json(folder, name):
    return json.loads((folder / name).read_text(encoding="utf-8"))


def gate_table(conditions: dict, gate_scores: dict, temperature: float) -> dict:
    header = f"{'condition':<22}{'mean_sim':>9}{'cov':>8}{'acc':>8}"
    for gate in GATES:
        header += f"{'cov@' + str(gate):>10}{'acc@' + str(gate):>10}"
    print(header)
    record = {}
    for name in EVIDENCE_CONDITIONS:
        rows = conditions[name]
        probs, labels, _ = load_rows(rows)
        calibrated = softmax_with_temperature(probs, temperature)
        scores = np.array([gate_scores[name][row["claim_id"]] for row in rows])
        cov, acc = coverage_and_accuracy(
            calibrated, labels, np.zeros(len(rows)), CONFIDENCE_THRESHOLD, 0.0
        )
        line = f"{name:<22}{scores.mean():>9.3f}{cov:>8.3f}{acc:>8.3f}"
        record[name] = {
            "mean_similarity": round(float(scores.mean()), 3),
            "no_gate": [round(cov, 3), round(acc, 3)],
        }
        for gate in GATES:
            cov, acc = coverage_and_accuracy(calibrated, labels, scores, CONFIDENCE_THRESHOLD, gate)
            line += f"{cov:>10.3f}{acc:>10.3f}"
            record[name][f"gate_{gate}"] = [round(cov, 3), round(acc, 3)]
        print(line)
    return record


def error_analysis(dev_rows, retrieval_top5, gold, claims, temperature) -> dict:
    probs, _, _ = load_rows(dev_rows)
    calibrated = softmax_with_temperature(probs, temperature)
    claim_text = dict(zip(claims["claim_id"], claims["claim"]))

    records = []
    for row, probability in zip(dev_rows, calibrated):
        claim_id = row["claim_id"]
        retrieved = {evidence_id for evidence_id, _ in retrieval_top5[claim_id]}
        gold_hits = len(retrieved & gold[claim_id])
        guess = LABELS[int(probability.argmax())]
        confidence = float(probability.max())
        records.append(
            {
                "claim_id": claim_id,
                "claim": claim_text[claim_id],
                "truth": row["label"],
                "prediction": guess,
                "confidence": round(confidence, 3),
                "gold_in_top5": gold_hits,
                "category": error_category(row["label"], guess, confidence, gold_hits),
            }
        )
    frame = pd.DataFrame(records)

    counts = Counter(frame["category"])
    print(f"\nerror categories on {len(frame)} dev claims:")
    for category, count in counts.most_common():
        print(f"  {category:<26}{count:>5}  ({count / len(frame):.1%})")
    for name, mask in (
        ("gold evidence retrieved", frame["gold_in_top5"] > 0),
        ("no gold evidence retrieved", frame["gold_in_top5"] == 0),
    ):
        part = frame[mask]
        print(
            f"  {name:<28} n={len(part):>4}  "
            f"accuracy={(part['truth'] == part['prediction']).mean():.3f}"
        )

    errors = frame[frame["category"] != "correct"].sort_values("category")
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    errors.to_csv(out_dir / "error_cases.csv", index=False)
    print(f"  {len(errors)} error cases written to outputs/error_cases.csv")
    return {category: int(count) for category, count in counts.items()}


def ablation(variants: dict, temperature: float) -> list:
    rows = [("claim only (no evidence)", *CLAIM_ONLY)]
    names = (
        ("bm25", "+ BM25 top-5"),
        ("dense", "+ dense top-5"),
        ("hybrid", "+ hybrid top-5"),
        ("dense_rerank_zero_shot", "+ dense + zero-shot reranker"),
    )
    for key, label in names:
        probs, labels, _ = load_rows(variants[key])
        predicted = probs.argmax(axis=1)
        accuracy = float((predicted == labels).mean())
        macro_f1 = f1_score(labels, predicted, labels=[0, 1, 2], average="macro", zero_division=0)
        rows.append((label, round(accuracy, 4), round(float(macro_f1), 4)))
    rows.append(("oracle: gold evidence", *GOLD_EVIDENCE))

    print(f"\n{'ablation (dev, 462 claims)':<32}{'accuracy':>10}{'macro_f1':>10}")
    for label, accuracy, macro_f1 in rows:
        print(f"{label:<32}{accuracy:>10.4f}{macro_f1:>10.4f}")

    probs, labels, _ = load_rows(variants["dense"])
    calibrated = softmax_with_temperature(probs, temperature)
    cov, acc = coverage_and_accuracy(
        calibrated, labels, np.zeros(len(labels)), CONFIDENCE_THRESHOLD, 0.0
    )
    print(
        f"{'+ calibration and abstention':<32} coverage={cov:.3f}  accuracy on answered={acc:.3f}"
    )
    return [list(row) for row in rows] + [
        ["calibrated abstention (dense)", round(cov, 3), round(acc, 3)]
    ]


def main() -> None:
    folder = PROJECT_ROOT / load_config()["paths"]["data_processed"] / "averitec"
    probs_data = read_json(folder, "verifier_probs_seed42.json")
    conditions = read_json(folder, "verifier_robustness_seed42.json")
    analysis = read_json(folder, "verifier_analysis_seed42.json")
    claims, evidence = load_averitec_tables()

    val_p, val_y, _ = load_rows(probs_data["val_retrieved"])
    temperature = fit_temperature(val_p, val_y)
    print(f"temperature {temperature:.2f}, confidence threshold {CONFIDENCE_THRESHOLD}\n")

    summary = {
        "experiment": "analysis_summary",
        "date": date.today().isoformat(),
        "gate": gate_table(conditions, analysis["gate_scores"], temperature),
        "errors": error_analysis(
            probs_data["dev_retrieved"],
            analysis["retrieval_top5"],
            gold_by_claim(evidence),
            claims,
            temperature,
        ),
        "ablation": ablation(analysis["variants"], temperature),
    }
    (PROJECT_ROOT / "experiments").mkdir(exist_ok=True)
    (PROJECT_ROOT / "experiments" / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
