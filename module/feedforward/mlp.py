import torch.nn as nn
from module.activation.gelu import GELU

class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.emb_dim, 4 * config.emb_dim),
            GELU() if config.activation == "gelu" else nn.ReLU(),
            nn.Linear(4 * config.emb_dim, config.emb_dim),
        )

    def forward(self,x):
        return self.layers(x)