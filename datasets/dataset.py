import os
import torch
import tiktoken
from torch.utils.data import Dataset

tokenizer = tiktoken.get_encoding("gpt2")

class VerdictDataset(Dataset):
    def __init__(self,raw_text,context_length,stride):
        super().__init__()
        tokenized_text = tokenizer.encode(raw_text)
        self.X = []
        self.y = []
        for i in range(0,len(tokenized_text)-context_length,stride):
            self.X.append(torch.tensor(tokenized_text[i:i+context_length]))
            self.y.append(torch.tensor(tokenized_text[i+1:i+context_length+1]))
        self.X = torch.stack(self.X)
        self.y = torch.stack(self.y)


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx],self.y[idx]

# file_path = os.path.join(os.getcwd(),'data','the-verdict', 'the-verdict.txt')
# verdict_dataset = VerdictDataset(file_path,4,1)
# print(len(verdict_dataset))
# for x,y in verdict_dataset:
#     print(len(x), len(y))