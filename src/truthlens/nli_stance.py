import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from truthlens.verification_data import LABELS

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
NLI_TO_STANCE = {"entailment": "SUPPORTED", "contradiction": "REFUTED", "neutral": "UNKNOWN"}


def reorder_nli_probs(probs: np.ndarray, id2label: dict) -> np.ndarray:
    """Reorder NLI columns (entailment/contradiction/neutral) into LABELS order."""
    column_of = {NLI_TO_STANCE[name.lower()]: int(i) for i, name in id2label.items()}
    return probs[:, [column_of[label] for label in LABELS]]


def load_nli_stance_fn(device: torch.device, model_name: str = NLI_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()

    def stance_probabilities(claim: str, snippets: list[str]) -> np.ndarray:
        batch = tokenizer(
            snippets,
            [claim] * len(snippets),
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**batch).logits, dim=-1).cpu().numpy()
        return reorder_nli_probs(probs, model.config.id2label)

    return stance_probabilities
