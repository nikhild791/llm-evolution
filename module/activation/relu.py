import torch
import torch.nn as nn

class ReLU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return torch.where(x>0,x,torch.zeros_like(x)) 
