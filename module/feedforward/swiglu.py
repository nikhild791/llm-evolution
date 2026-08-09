import torch.nn as nn
from activation.silu import SiLU

class SwiGLU(nn.Module):
    def __init__(self,config):
        self.gate = nn.Linear(config.emb_dim,config.hidden_dim, bias=False)
        self.proj = nn.Linear(config.emb_dim, config.hidden_dim, bias=False)
        self.up = nn.Linear(config.hidden_dim, config.emb_dim, bias=False)
        self.activation = SiLU()

    def forward(self,x):
        gate = self.activation(self.gate(x))
        up = self.up(x)
        return self.proj(gate*up)
 