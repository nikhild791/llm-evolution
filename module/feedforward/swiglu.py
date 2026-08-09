import torch.nn as nn
from ..activation.silu import SiLU

class SwiGLU(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.gate_proj = nn.Linear(config.emb_dim,config.hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.emb_dim, config.hidden_dim, bias=False)
        self.down_proj = nn.Linear(config.hidden_dim, config.emb_dim, bias=False)
        self.activation = SiLU()

    def forward(self,x):
        gate = self.activation(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate*up)
 