from dataclasses import dataclass

@dataclass
class SchedulerConfig:
    name: str = "cosine"
    warmup_steps: int = 1000
    step_size: int = 100000
    gamma: float = 0.01