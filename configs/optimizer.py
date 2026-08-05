from dataclasses import dataclass

@dataclass
class OptimizerConfig:
    optimizer: str = "AdamW"
    lr: float = 3e-4
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1