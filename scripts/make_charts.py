import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from truthlens.config import PROJECT_ROOT

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a8a7a2"
# 3-seed means from the ablation logs; claim-only and gold are not stored in seed_summary.json
CLAIM_ONLY = (0.647, 0.013)
GOLD_EVIDENCE = (0.729, 0.004)
MAJORITY_ACCURACY = 0.660

EXPERIMENTS = PROJECT_ROOT / "experiments"
FIGURES = PROJECT_ROOT / "docs" / "figures"


def load(name: str) -> dict:
    return json.loads((EXPERIMENTS / f"{name}.json").read_text(encoding="utf-8"))


def new_axes(title: str, subtitle: str, size=(9, 4.8)):
    fig, ax = plt.subplots(figsize=size, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, labelsize=10)
    fig.text(0.02, 0.96, title, fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.895, subtitle, fontsize=10, color=INK_2, va="top")
    fig.subplots_adjust(top=0.80, left=0.30, right=0.95, bottom=0.12)
    return fig, ax


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def ablation_chart() -> None:
    variants = load("seed_summary")["variants"]
    rows = [("claim only (no evidence)", *CLAIM_ONLY, GRAY)]
    for key, label in (
        ("bm25", "BM25 evidence"),
        ("dense", "dense evidence"),
        ("hybrid", "hybrid evidence (used)"),
        ("dense_rerank_zero_shot", "dense + reranker"),
    ):
        values = variants[key]["accuracy"]
        rows.append((label, float(np.mean(values)), float(np.std(values)), BLUE))
    rows.append(("gold evidence (ceiling)", *GOLD_EVIDENCE, GRAY))

    fig, ax = new_axes(
        "Evidence helps, but only a few points",
        "Verifier accuracy on 462 dev claims, mean +- std over 3 seeds. Dashed line = always "
        "answer REFUTED.",
    )
    y = np.arange(len(rows))[::-1]
    for yi, (label, mean, std, color) in zip(y, rows, strict=True):
        ax.barh(yi, mean, height=0.5, color=color)
        ax.errorbar(mean, yi, xerr=std, color=INK_2, capsize=3, linewidth=1)
        ax.text(mean + std + 0.004, yi, f"{mean + 1e-9:.3f}", va="center", fontsize=10, color=INK)
    ax.axvline(MAJORITY_ACCURACY, color=INK_2, linestyle="--", linewidth=1)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0.55, 0.78)
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    save(fig, "ablation_accuracy")


def robustness_chart() -> None:
    conditions = load("robustness")["conditions"]
    labels = {
        "clean": "clean",
        "claim_typos": "claim typos",
        "claim_noisy_text": "noisy text",
        "claim_sensational": "sensational wording",
        "mixed_evidence": "3 real + 2 misleading",
        "random_evidence": "random evidence",
        "misleading_evidence": "misleading evidence",
        "no_evidence": "no evidence",
    }
    bad = {"random_evidence", "misleading_evidence", "no_evidence"}
    fig, ax = new_axes(
        "Robust to typos and noise, fooled by bad evidence",
        "Dev accuracy per condition (seed 42). Orange = evidence is wrong, missing or random.",
    )
    y = np.arange(len(labels))[::-1]
    for yi, (key, label) in zip(y, labels.items(), strict=True):
        value = conditions[key]["accuracy"]
        ax.barh(yi, value, height=0.5, color=ORANGE if key in bad else BLUE)
        ax.text(value + 0.01, yi, f"{value:.2f}", va="center", fontsize=10, color=INK)
    ax.set_yticks(y, list(labels.values()))
    ax.set_xlim(0, 0.85)
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    save(fig, "robustness_accuracy")


def abstention_chart() -> None:
    points = load("operating_points")["points"]
    argmax = load("decision_calibration")["dev"]["argmax (abstain only if UNKNOWN)"]
    rows = [("answer almost everything", argmax["coverage"], argmax["accuracy_on_answered"])]
    names = {"0.75": "target 0.75", "0.8": "target 0.80 (used)", "0.85": "target 0.85"}
    for key, name in names.items():
        dev = points[key]["dev"]
        rows.append((name, dev["coverage"], dev["accuracy"]))
    rows.sort(key=lambda r: -r[1])

    fig, ax = new_axes(
        "Answering fewer claims raises accuracy, up to a point",
        "Dev set (seed 42): claims answered vs accuracy on answered ones. 0.85 target missed.",
    )
    fig.subplots_adjust(left=0.10)
    ax.plot([r[1] for r in rows], [r[2] for r in rows], color=INK_2, linewidth=1.5, zorder=1)
    offsets = {
        "answer almost everything": (-8, 12, "right"),
        "target 0.75": (8, 12, "left"),
        "target 0.80 (used)": (0, 14, "center"),
        "target 0.85": (0, -34, "center"),
    }
    for label, coverage, accuracy in rows:
        ax.scatter(coverage, accuracy, s=70, color=BLUE, zorder=2, edgecolor=SURFACE, linewidth=2)
        dx, dy, align = offsets[label]
        ax.annotate(
            f"{label}\n{coverage:.0%} answered, {accuracy:.1%} correct",
            (coverage, accuracy),
            textcoords="offset points",
            xytext=(dx, dy),
            ha=align,
            fontsize=9,
            color=INK,
        )
    ax.set_xlabel("coverage (share of claims answered)", color=INK_2)
    ax.set_ylabel("accuracy on answered claims", color=INK_2)
    ax.set_xlim(0.2, 1.1)
    ax.set_ylim(0.68, 0.90)
    ax.grid(True, color=GRID)
    ax.set_axisbelow(True)
    save(fig, "abstention_tradeoff")


def live_chart() -> None:
    runs = load("live_eval_summary")["runs"]
    names = [
        "verifier, basic",
        "NLI, basic",
        "NLI, advanced\n(2-domain REFUTED)",
        "NLI, advanced\n(final rules)",
    ]
    fig, ax = new_axes(
        "Live web verification: safe but rarely answers",
        "20 custom claims, one run each. Settings were tuned on the same claims, so optimistic.",
    )
    fig.subplots_adjust(left=0.22)
    y = np.arange(len(runs))[::-1]
    for yi, run in zip(y, runs, strict=True):
        wrong = run["wrong_answers"]
        ax.barh(yi, run["coverage"], height=0.5, color=ORANGE if wrong else BLUE)
        ax.text(
            run["coverage"] + 0.01,
            yi,
            f"{run['answered']} of 20 answered, {wrong} wrong",
            va="center",
            fontsize=10,
            color=INK,
        )
    ax.set_yticks(y, names)
    ax.set_xlim(0, 0.6)
    ax.set_xlabel("coverage (share of claims answered)", color=INK_2)
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    save(fig, "live_coverage")


def main() -> None:
    ablation_chart()
    robustness_chart()
    abstention_chart()
    live_chart()
    print(f"saved charts to {FIGURES}")


if __name__ == "__main__":
    main()
