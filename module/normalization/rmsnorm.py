import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self,vec_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(vec_dim))
        self.eps = 1e-6

    def forward(self,x):
        rms = torch.rsqrt(torch.mean(x**2,dim=-1,keepdim=True)+self.eps)
        return self.weight*x*rms