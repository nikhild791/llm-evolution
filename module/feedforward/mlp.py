import torch.nn as nn
from module.activation.gelu import GELU
from linearlayer import LinearLayer

class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.layers = nn.Sequential(
            LinearLayer(config.emb_dim,  config.hidden_dim),
            GELU() if config.activation == "gelu" else nn.ReLU(),
            LinearLayer(4 * config.hidden_dim, config.emb_dim),
        )

    def forward(self,x):
        return self.layers(x)