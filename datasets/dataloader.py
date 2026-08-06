import tiktoken
from torch.utils.data import DataLoader

from .dataset import VerdictDataset


def verdictDataLoader(text, batch_size=32,  context_length=32, stride=32, shuffle=False,tokenizer=None, num_workers=0, drop_last=True):
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("gpt2")

    train_config = type("TrainConfig", (), {
        "context_length": context_length,
        "stride": stride,
    })()

    dataset = VerdictDataset(text, tokenizer, train_config)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )