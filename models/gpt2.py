import torch
import torch.nn as nn
import torch.nn.functional as F

from module.attention.mha import MHA
from module.activation.gelu import GELU
from module.normalization.layernorm import LayerNorm
from module.feedforward.mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mha =MHA(config)
        self.ln1 = LayerNorm(config)
        self.ln2 = LayerNorm(config)
        self.dropout = nn.Dropout(config.dropout)
        self.ffn = MLP(config)

    def forward(self,x):
        shortcut = x
        x = self.ln1(x)
        x = self.mha(x)
        x = self.dropout(x)
        x = x + shortcut          

        shortcut = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + shortcut          
        return x
        

class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.emb_dim)
        self.pte = nn.Embedding(config.context_length, config.emb_dim)
        self.trf_blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layers)])
        self.finalNorm = nn.LayerNorm(config.emb_dim)
        self.final_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)
        
    
    def forward(self,x):
        _,T = x.shape
        tok_enc = self.wte(x)
        pos_ids = torch.arange(T, device=x.device)
        pos_enc = self.pte(pos_ids)
        x = tok_enc + pos_enc
        logits = self.trf_blocks(x)
        logits = self.finalNorm(logits)
        logits = self.final_head(logits)
        return logits



