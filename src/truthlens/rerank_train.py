import random

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


class PairScorer:
    def __init__(self, model, tokenizer, device, max_length: int = 256):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    @torch.no_grad()
    def predict(self, pairs: list[list[str]], batch_size: int = 64, **kwargs) -> list[float]:
        self.model.eval()
        scores = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            encoded = self.tokenizer(
                [pair[0] for pair in batch],
                [pair[1] for pair in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            scores.extend(self.model(**encoded).logits.squeeze(-1).float().cpu().tolist())
        return scores


def fine_tune_reranker(
    model_name: str,
    pairs: list[list[str]],
    labels: list[int],
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    max_length: int = 256,
) -> PairScorer:
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps_per_epoch = (len(pairs) + batch_size - 1) // batch_size
    total_steps = epochs * steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss()

    order = list(range(len(pairs)))
    rng = random.Random(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(order)
        running_loss = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            encoded = tokenizer(
                [pairs[i][0] for i in batch],
                [pairs[i][1] for i in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            target = torch.tensor([labels[i] for i in batch], dtype=torch.float, device=device)

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**encoded).logits.squeeze(-1)
            loss = loss_fn(logits.float(), target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()
        print({"epoch": epoch, "train_loss": round(running_loss / steps_per_epoch, 4)})

    return PairScorer(model, tokenizer, device, max_length)
