import pandas as pd

LABEL_MAP = {
    "Supported": "SUPPORTED",
    "Refuted": "REFUTED",
    "Not Enough Evidence": "UNKNOWN",
    "Conflicting Evidence/Cherrypicking": "CONFLICTING",
}


def build_tables(claims: list[dict], split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    claim_rows, evidence_rows = [], []
    for claim_index, claim in enumerate(claims):
        claim_id = f"{split}-{claim_index}"
        n_evidence = 0
        for q_index, question in enumerate(claim["questions"]):
            for a_index, answer in enumerate(question["answers"]):
                if answer.get("answer_type") == "Unanswerable":
                    continue
                n_evidence += 1
                evidence_rows.append(
                    {
                        "evidence_id": f"{claim_id}-{q_index}-{a_index}",
                        "claim_id": claim_id,
                        "question": question["question"],
                        "answer": answer.get("answer", ""),
                        "answer_type": answer["answer_type"],
                        "source_url": answer.get("source_url"),
                        "text": f"Q: {question['question']} A: {answer.get('answer', '')}",
                    }
                )
        claim_rows.append(
            {
                "claim_id": claim_id,
                "split": split,
                "claim": claim["claim"],
                "label": claim["label"],
                "label_mapped": LABEL_MAP[claim["label"]],
                "claim_date": claim.get("claim_date"),
                "n_evidence": n_evidence,
            }
        )
    return pd.DataFrame(claim_rows), pd.DataFrame(evidence_rows)
