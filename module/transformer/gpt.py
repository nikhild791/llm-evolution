import torch.nn as nn

from module.attention.mha import MHA
from module.normalization.layernorm import LayerNorm
from module.feedforward.mlp import MLP

class GPTTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention =MHA(config)
        self.ln1 = LayerNorm(config)
        self.ln2 = LayerNorm(config)
        self.dropout = nn.Dropout(config.dropout)
        self.ffn = MLP(config)

    def forward(self,x):
        shortcut = x
        x = self.ln1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + shortcut          

        shortcut = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + shortcut          
        return x