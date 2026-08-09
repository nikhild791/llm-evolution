from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    context_length: int = 1024
    emb_dim: int = 768
    hidden_dim: int = 4*768
    n_layers: int = 12
    n_heads: int = 12
    kv_heads: int = 4
    dropout: float = 0.1
    activation:str = "relu"
    qkv_bias: bool = False
