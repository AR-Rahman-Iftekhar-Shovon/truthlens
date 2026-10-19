import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup

from truthlens.evaluation import summarize


def set_torch_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device) -> list[int]:
    model.eval()
    predictions = []
    for batch in loader:
        inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
        predictions.extend(model(**inputs).logits.argmax(dim=-1).cpu().tolist())
    return predictions


def fine_tune(model, train_ds, val_ds, tokenizer, id_to_label, epochs, lr, batch_size, seed):
    device = torch.device("cuda")
    model.to(device)
    collator = DataCollatorWithPadding(tokenizer)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator, generator=generator
    )
    val_loader = DataLoader(val_ds, batch_size=64, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda")

    labels = [id_to_label[i] for i in sorted(id_to_label)]
    val_true = [id_to_label[i] for i in val_ds.labels]
    history, best_summary, best_state = [], None, None

    for epoch in range(1, epochs + 1):
        model.train()
        start = time.time()
        running_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()

        predicted = [id_to_label[i] for i in predict(model, val_loader, device)]
        summary = summarize(val_true, predicted, labels)
        entry = {
            "epoch": epoch,
            "train_loss": round(running_loss / len(train_loader), 4),
            "accuracy": summary["accuracy"],
            "macro_f1": summary["macro_f1"],
            "seconds": round(time.time() - start),
        }
        history.append(entry)
        print(entry)

        if best_summary is None or summary["macro_f1"] > best_summary["macro_f1"]:
            best_summary = summary
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

    return history, best_summary, best_state
