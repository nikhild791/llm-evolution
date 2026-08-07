from dataclasses import dataclass

@dataclass
class CheckpointConfig:
    save_every_steps: int = 96
    save_best: bool = True
    keep_last_n: int = 3
    path : str = 'checkpoints'