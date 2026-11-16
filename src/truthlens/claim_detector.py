import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from truthlens.config import PROJECT_ROOT
from truthlens.transformer_data import StatementDataset
from truthlens.transformer_train import predict_proba

MODEL = "distilroberta-base"
MAX_LENGTH = 128
CHECK_WORTHY_ID = 1


def load_detector(seed: int = 42) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    weights = PROJECT_ROOT / "outputs" / f"distilroberta_base_claimbuster_seed{seed}.pt"
    model.load_state_dict(torch.load(weights))
    device = torch.device("cuda")
    return {"tokenizer": tokenizer, "model": model.to(device), "device": device}


def score_sentences(sentences: list[str], detector: dict) -> list[float]:
    """Probability that each sentence is check-worthy."""
    tokenizer = detector["tokenizer"]
    dataset = StatementDataset(sentences, [0] * len(sentences), tokenizer, MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=32, collate_fn=DataCollatorWithPadding(tokenizer))
    probabilities = np.array(predict_proba(detector["model"], loader, detector["device"]))
    return probabilities[:, CHECK_WORTHY_ID].tolist()
