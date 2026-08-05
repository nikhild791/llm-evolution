from dataclasses import dataclass

@dataclass
class GenerationConfig:
    max_new_tokens: int = 200
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95