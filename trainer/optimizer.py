import torch

def build_optimizer(model, optimizer_config ):
    return torch.optim.AdamW(
    model.parameters(),
    lr=optimizer_config.lr,
    weight_decay=optimizer_config.weight_decay
)