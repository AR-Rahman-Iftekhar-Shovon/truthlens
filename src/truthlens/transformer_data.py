import torch


class StatementDataset(torch.utils.data.Dataset):
    def __init__(self, statements: list[str], labels: list[int], tokenizer, max_length: int):
        self.encodings = tokenizer(statements, truncation=True, max_length=max_length)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        item = {key: values[index] for key, values in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item
