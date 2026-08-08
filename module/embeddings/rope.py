import math
import torch
import torch.nn as nn

class RotaryPE(nn.Module):
    def __init__(self,config):
        super().__init__()
        positions = torch.arange(config.context_length)
        div_term = torch.exp(torch.arange(0,config.head_dim,2)*(-math.log(10000)/config.head_dim))
        sin = torch.sin(positions*div_term)
        cos = torch.cos(positions*div_term)
        self.register_buffer('sin', sin)
        self.register_buffer('cos', cos)

    def forward(self, x):
            _,_,T,_ = x.shape      ###  B,num_heads,T,head_dim
            sin = self.sin[:T].unsqueeze(0).unsqueeze(0)        ### (1, 1, T, head_dim/2)
            cos = self.cos[:T].unsqueeze(0).unsqueeze(0)

            x_even = x[:, :, :, 0::2]   ### (B, num_head, T, head_dim/2)
            x_odd = x[:, :, :, 1::2]
            rotate_even = (
                x_even*cos 
               - x_odd*sin
            )
           
            rotate_odd = (
                x_even*sin
                + x_odd*cos
            )
            output = torch.stack(
                [rotate_even, rotate_odd],
                dim=-1
            ).flatten(-2)
           
            return output

### while learning i have implemented rotary on token embeding dimesions but learned that they are commonly applied to
### attention heads here is code to learn when i forget i 

# class RotaryPE(nn.Module):
#     def __init__(self,config):
#         super().__init__()
#         positions = torch.arange(config.context_length)
#         div_term = torch.exp(torch.arange(0,config.emb_dim,2)*(-math.log(10000)/config.emb_dim))
#         sin = torch.sin(positions*div_term)
#         cos = torch.cos(positions*div_term)
#         self.register_buffer('sin', sin)
#         self.register_buffer('cos', cos)

#     def forward(self, x):
#         if len(x.shape) == 3:     ### x has size B,T,emb_dim we apply rotation on token embedding
#             _,T,_ = x.shape
#             sin = self.sin[:T]
#             cos = self.cos[:T]

#             x_even = x[:, :, 0::2]
#             x_odd = x[:, :, 1::2]
#             rotate_even = (
#                 x_even*cos 
#                 - x_odd*sin
#             )

#             rotate_odd = (
#                 x_even*sin
#                 + x_odd*cos
#             )

#             output = torch.zeros_like(x)
#             output[:,:,0::2] = rotate_even
#             output[:,:,1::2] = rotate_odd

#             return output

#         if len(x.shape) == 4:      ### x has size B,num_heads,T,head_dim now we have to apply rotation on att heads
#             _,_,T,head_dim = x.shape      ###  B,num_heads,T,head_dim
#             sin = self.sin[:T][:head_dim//2]
#             cos = self.cos[:T][:head_dim//2]

#             x_even = x[:, :, :, 0::2]
#             x_odd = x[:, :, :, 1::2]
#             rotate_even = (
#                 x_even*cos 
#                - x_odd*sin
#             )
           
#             rotate_odd = (
#                 x_even*sin
#                 + x_odd*cos
#             )
#             output = torch.zeros_like(x)
#             output[:,:,:,0::2] = rotate_even
#             output[:,:,:,1::2] = rotate_odd
           
#             return output