from dataclasses import dataclass

@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 32
    stride: int = 32
    context_length: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    mixed_precision: bool = False
    shuffle: bool = False
    gradient_accumulation_steps: int = 1