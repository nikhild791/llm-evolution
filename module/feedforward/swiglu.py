import torch.nn as nn
from ..activation.silu import SiLU

class SwiGLU(nn.Module):
    def __init__(self,input,output,bias):
        super().__init__()
        self.gate_proj = nn.Linear(input,output, bias=bias)
        self.up_proj = nn.Linear(input, output, bias=bias)
        self.down_proj = nn.Linear(output, input, bias=bias)
        self.activation = SiLU()

    def forward(self,x):
        gate = self.activation(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate*up)
 