import torch.nn as nn
from module.transformer.gpt import GPTTransformerBlock
from module.embeddings.learned import LearnedPE
from module.embeddings.tokenEmb import TokenEmbedding
from module.normalization.layernorm import LayerNorm
from module.feedforward.linearlayer import LinearLayer
        

class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.wte = TokenEmbedding(config)
        self.pte = LearnedPE(config)
        self.trf_blocks = nn.Sequential(*[GPTTransformerBlock(config) for _ in range(config.n_layers)])
        self.finalNorm = LayerNorm(config)
        self.final_head = LinearLayer(config.emb_dim, config.vocab_size, bias=False)
        
    
    def forward(self,x):
        tok_enc = self.wte(x)
        pos_enc = self.pte(x)
        x = tok_enc + pos_enc
        logits = self.trf_blocks(x)
        logits = self.finalNorm(logits)
        logits = self.final_head(logits)
        return logits



