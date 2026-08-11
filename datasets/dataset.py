import torch
from torch.utils.data import Dataset


class VerdictDataset(Dataset):
    def __init__(self, tokenized_text,  train_config):
        super().__init__()
        self.X = []
        self.y = []
        # print(len(tokenized_text))       ### check total number of token

        if len(tokenized_text) <= train_config.context_length:
            raise ValueError("tokenized text must be longer than context length")

        for i in range(0, len(tokenized_text) - train_config.context_length, train_config.stride):
            self.X.append(torch.tensor(tokenized_text[i : i + train_config.context_length],dtype=torch.long))
            self.y.append(torch.tensor(tokenized_text[i + 1 : i + train_config.context_length + 1],dtype=torch.long))

        if not self.X:
            raise ValueError("no training windows were created")

        self.X = torch.stack(self.X)
        self.y = torch.stack(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
