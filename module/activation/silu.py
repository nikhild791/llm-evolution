import torch
import torch.nn as nn

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x/(1 + torch.exp(-x))