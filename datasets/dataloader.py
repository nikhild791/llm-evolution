import tiktoken
from torch.utils.data import DataLoader
from .dataset import VerdictDataset

def create_dataloader(text,train_config):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = VerdictDataset(text,tokenizer,train_config)
    return DataLoader(dataset, batch_size=train_config.batch_size, shuffle=train_config.shuffle, drop_last=train_config.drop_last, num_workers=train_config.num_workers)