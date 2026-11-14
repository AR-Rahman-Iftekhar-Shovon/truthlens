import random

import pandas as pd

from truthlens.verification_data import build_examples


def add_typos(text: str, rate: float, seed: int) -> str:
    rng = random.Random(f"{seed}-{text}")
    words = text.split()
    for index, word in enumerate(words):
        if len(word) > 3 and rng.random() < rate:
            position = rng.randrange(len(word) - 1)
            words[index] = (
                word[:position] + word[position + 1] + word[position] + word[position + 2 :]
            )
    return " ".join(words)


def make_sensational(text: str) -> str:
    return f"BREAKING!!! SHOCKING: {text.upper()}!!!"


def strip_noise(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum() or ch.isspace())


def evidence_conditions(
    retrieved_texts: dict[str, list[str]],
    misleading: dict[str, list[str]],
    evidence: pd.DataFrame,
    seed: int,
    max_passages: int = 5,
) -> dict[str, dict[str, list[str]]]:
    rng = random.Random(seed)
    pool = list(zip(evidence["claim_id"], evidence["text"]))
    random_texts = {}
    for claim_id in retrieved_texts:
        others = [text for owner, text in pool if owner != claim_id]
        random_texts[claim_id] = rng.sample(others, max_passages)
    return {
        "clean": retrieved_texts,
        "misleading_evidence": {
            claim_id: misleading.get(claim_id) or texts
            for claim_id, texts in retrieved_texts.items()
        },
        "mixed_evidence": {
            claim_id: texts[:3] + misleading.get(claim_id, [])[:2]
            for claim_id, texts in retrieved_texts.items()
        },
        "random_evidence": random_texts,
    }


def build_conditions(
    dev_claims: pd.DataFrame,
    retrieved_texts: dict[str, list[str]],
    misleading: dict[str, list[str]],
    evidence: pd.DataFrame,
    seed: int,
    max_passages: int = 5,
) -> dict[str, pd.DataFrame]:
    texts = evidence_conditions(retrieved_texts, misleading, evidence, seed, max_passages)
    conditions = {"clean": build_examples(dev_claims, texts["clean"], max_passages)}

    perturbations = (
        ("claim_typos", lambda text: add_typos(text, 0.15, seed)),
        ("claim_sensational", make_sensational),
        ("claim_noisy_text", strip_noise),
    )
    for name, transform in perturbations:
        perturbed = dev_claims.assign(claim=dev_claims["claim"].map(transform))
        conditions[name] = build_examples(perturbed, retrieved_texts, max_passages)

    for name in ("misleading_evidence", "mixed_evidence", "random_evidence"):
        conditions[name] = build_examples(dev_claims, texts[name], max_passages)
    conditions["no_evidence"] = build_examples(dev_claims, {}, max_passages)
    return conditions
