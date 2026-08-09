import torch.nn as nn

from module.normalization.rmsnorm import RMSNorm
from module.attention.gqa import GQA
from module.feedforward.swiglu import SwiGLU

class QwenTransformerBlock(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.rmsnorm1 = RMSNorm(config.emb_dim)
        self.attention = GQA(config)
        self.rmsnorm2 = RMSNorm(config.emb_dim)
        self.swiglu = SwiGLU(config)


    def forward(self,x):
        shortcut = x
        x = self.rmsnorm1(x)
        x = self.attention(x)
        x = x +  shortcut

        shortcut = x
        x = self.rmsnorm2(x)
        x = self.swiglu(x)
        x = x + shortcut
        return x