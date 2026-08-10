import torch
import torch.nn as nn

from ..embeddings.rope import RotaryPE

class MLA(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.emb_dim// config.n_heads
        assert config.emb_dim % config.n_heads == 0
        self.latent_dim = config.latent_dim

        self.W_q = nn.Linear(config.emb_dim, config.emb_dim, bias=config.qkv_bias)
        self.W_dkv = nn.Linear(config.emb_dim, config.latent_dim, bias=config.qkv_bias)
        self.W_uk = nn.Linear(config.latent_dim, config.emb_dim, bias=config.qkv_bias)
        self.W_uv = nn.Linear(config.latent_dim, config.emb_dim, bias=config.qkv_bias) 
        self.W_o = nn.Linear(config.emb_dim, config.emb_dim, bias=config.qkv_bias)    
        self.register_buffer('mask', torch.triu(torch.ones(config.context_length, config.context_length,dtype=torch.bool),diagonal=1))   

        self.rotary = RotaryPE(config)

    def forward(self,x):
        B,T,emb_dim = x.shape
        C_kv = self.W_dkv(x)
        Q = self.W_q(x)
        K = self.W_uk(C_kv)
        V = self.W_uv(C_kv)

        ### unrolling Q,K,V =>b,t,emb_dim => b,t,n_heads,head_dim
        Q = Q.view(B,T,self.n_heads, self.head_dim)
        K = K.view(B,T,self.n_heads, self.head_dim)
        V = V.view(B,T,self.n_heads, self.head_dim)

        ### Q,K b,t,n_heads,head_dim => b,n_heads,t,head_dim
        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)

        ### rotation of Q, K
        Q = self.rotary(Q)
        K = self.rotary(K)

        att_score = Q@K.transpose(-2,-1)
        att_score = att_score/(self.head_dim)**0.5
        att_score = att_score.masked_fill(self.mask[:T,:T].bool(), float('-inf'))
        att_weight = torch.softmax(att_score,dim=-1)
        context_vec = att_weight @ V
        context_vec = context_vec.transpose(1,2)
        context_vec = context_vec.contiguous().view(B,T,emb_dim)
        return context_vec   