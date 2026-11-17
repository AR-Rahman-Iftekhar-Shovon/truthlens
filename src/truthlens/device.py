import torch


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
