import math
import torch
import torch.nn as nn


class SinoduidalPE(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.emb_dim = config.emb_dim
        positions = torch.arange(config.context_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,config.emb_dim,2)*(-math.log(10000.0)/config.emb_dim))
        pe = torch.zeros(config.context_length,config.emb_dim)
        pe[:,::2] = torch.sin(positions*div_term)
        pe[:,1::2] = torch.cos(positions*div_term)
        self.register_buffer('pe',pe)
         

    def forward(self,x):
        _,T =x.shape
        # batch_pos = (self.pe[:T]).repeat(B,1,1) ### we dont need to repeat cz 1,T,emb will be broadcast when added token B,T,emb
        batch_pos = self.pe[:T]
        return batch_pos
    


