import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim)

    def forward(self,x):
        return self.tok_emb(x)