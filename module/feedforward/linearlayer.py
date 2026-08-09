import torch.nn as nn

class LinearLayer(nn.Module):
    def __init__(self,inp,out,bias):
        super().__init__()
        self.nn = nn.Linear(inp,out,bias=bias)

    def forward(self,x):
        return self.nn(x)