import torch.nn as nn

from module.attention.mla import MLA
from module.normalization.rmsnorm import RMSNorm
from module.feedforward.moe import MoE

class DeepSeekTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.moe = MoE(config)
        self.rmsnorm1 = RMSNorm(config.emb_dim)
        self.attention = MLA(config)
        self.rmsnorm2 = RMSNorm(config.emb_dim)

    def forward(self,x):
        shortcut = x
        x = self.rmsnorm1(x)
        x = self.attention(x)
        x = x +  shortcut

        shortcut = x
        x = self.rmsnorm2(x)
        x = self.moe(x)
        x = x + shortcut
        return x