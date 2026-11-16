import numpy as np


def drop_one(passages: list[str]) -> list[list[str]]:
    return [passages[:i] + passages[i + 1 :] for i in range(len(passages))]


def claim_faithfulness(
    full: np.ndarray, no_evidence: np.ndarray, dropped: np.ndarray, is_gold: list[bool]
) -> dict:
    """Probabilities with all passages, with none, and with each passage removed in turn."""
    predicted = int(full.argmax())
    drops = [float(full[predicted] - probs[predicted]) for probs in dropped]
    flips = [int(probs.argmax()) != predicted for probs in dropped]
    most_influential = int(np.argmax(drops))
    return {
        "no_evidence_changes": int(no_evidence.argmax()) != predicted,
        "max_drop": max(drops),
        "any_flip": any(flips),
        "top_is_gold": is_gold[most_influential],
        "n_gold": sum(is_gold),
        "n_passages": len(is_gold),
        "drops_gold": [d for d, g in zip(drops, is_gold, strict=True) if g],
        "drops_other": [d for d, g in zip(drops, is_gold, strict=True) if not g],
    }


def summarize_faithfulness(rows: list[dict]) -> dict:
    mixed = [r for r in rows if 0 < r["n_gold"] < r["n_passages"]]
    gold_drops = [d for r in rows for d in r["drops_gold"]]
    other_drops = [d for r in rows for d in r["drops_other"]]
    return {
        "n_claims": len(rows),
        "no_evidence_changes_verdict": float(np.mean([r["no_evidence_changes"] for r in rows])),
        "any_single_removal_flips_verdict": float(np.mean([r["any_flip"] for r in rows])),
        "mean_max_probability_drop": float(np.mean([r["max_drop"] for r in rows])),
        "claims_with_gold_and_other_passages": len(mixed),
        "top_influential_is_gold": float(np.mean([r["top_is_gold"] for r in mixed])),
        "chance_top_is_gold": float(np.mean([r["n_gold"] / r["n_passages"] for r in mixed])),
        "mean_drop_gold_passage": float(np.mean(gold_drops)) if gold_drops else None,
        "mean_drop_other_passage": float(np.mean(other_drops)) if other_drops else None,
    }
