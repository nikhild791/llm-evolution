import torch
import torch.nn as nn

class LearnedPE(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.pos = nn.Embedding(config.context_length, config.emb_dim)

    def forward(self,x):
        _,T = x.shape
        pos = self.pos(torch.arange(T, device=x.device))
        return pos