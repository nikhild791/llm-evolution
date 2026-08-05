
import math
import torch
import torch.nn as nn

class MHA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.emb_dim // config.n_heads
        self.n_heads = config.n_heads
        self.W_q = nn.Linear(config.emb_dim, config.emb_dim, bias=config.qkv_bias)
        self.W_k = nn.Linear(config.emb_dim, config.emb_dim, bias=config.qkv_bias)
        self.W_v = nn.Linear(config.emb_dim, config.emb_dim, bias=config.qkv_bias)
        self.out_proj = nn.Linear(config.emb_dim, config.emb_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer('mask', torch.triu(torch.ones(config.context_length, config.context_length), diagonal=1))

    def forward(self,X):
        b,T,emb_dim = X.shape
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        ### unroll the Q,K,V from b,T,emb_dim => b,T,n_heads,head_dim
        Q = Q.view(b,T,self.n_heads, self.head_dim)
        K = K.view(b,T,self.n_heads, self.head_dim)
        V = V.view(b,T,self.n_heads, self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        att = Q @ K.transpose(-2, -1)
        att = att / math.sqrt(self.head_dim)

        att = att.masked_fill(
            self.mask[:T, :T].bool(),
            float("-inf")
        )

        att = torch.softmax(att, dim=-1)

        context = att @ V
        context = context.transpose(1, 2).contiguous()
        context = context.view(b, T, emb_dim)
        context = self.out_proj(context)
        return context