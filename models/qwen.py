import torch.nn as nn
from module.transformer.qwen import QwenTransformerBlock
from module.embeddings.tokenEmb import TokenEmbedding
from module.normalization.rmsnorm import RMSNorm
from module.feedforward.linearlayer import LinearLayer
        

class Qwen(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.wte = TokenEmbedding(config)
        self.trf_blocks = nn.Sequential(*[QwenTransformerBlock(config) for _ in range(config.n_layers)])
        self.finalNorm = RMSNorm(config.emb_dim)
        self.final_head = LinearLayer(config.emb_dim, config.vocab_size, bias=False)
        
    
    def forward(self,x):
        tok_enc = self.wte(x)
        logits = self.trf_blocks(tok_enc)
        logits = self.finalNorm(logits)
        logits = self.final_head(logits)
        return logits
