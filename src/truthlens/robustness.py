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


def build_conditions(
    dev_claims: pd.DataFrame,
    retrieved_texts: dict[str, list[str]],
    misleading: dict[str, list[str]],
    evidence: pd.DataFrame,
    seed: int,
    max_passages: int = 5,
) -> dict[str, pd.DataFrame]:
    rng = random.Random(seed)
    conditions = {"clean": build_examples(dev_claims, retrieved_texts, max_passages)}

    perturbations = (
        ("claim_typos", lambda text: add_typos(text, 0.15, seed)),
        ("claim_sensational", make_sensational),
        ("claim_noisy_text", strip_noise),
    )
    for name, transform in perturbations:
        perturbed = dev_claims.assign(claim=dev_claims["claim"].map(transform))
        conditions[name] = build_examples(perturbed, retrieved_texts, max_passages)

    only_misleading = {cid: misleading.get(cid) or texts for cid, texts in retrieved_texts.items()}
    conditions["misleading_evidence"] = build_examples(dev_claims, only_misleading, max_passages)

    mixed = {cid: texts[:3] + misleading.get(cid, [])[:2] for cid, texts in retrieved_texts.items()}
    conditions["mixed_evidence"] = build_examples(dev_claims, mixed, max_passages)

    pool = list(zip(evidence["claim_id"], evidence["text"]))
    random_texts = {}
    for claim_id in retrieved_texts:
        others = [text for owner, text in pool if owner != claim_id]
        random_texts[claim_id] = rng.sample(others, max_passages)
    conditions["random_evidence"] = build_examples(dev_claims, random_texts, max_passages)

    conditions["no_evidence"] = build_examples(dev_claims, {}, max_passages)
    return conditions
