import torch
def generate_sample_text(logits, tokenizer):
    output = torch.argmax(logits, dim=-1)
    