import torch

def build_scheduler(optimizer,scheduler_config):
    return torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=scheduler_config.step_size,
    gamma= scheduler_config.gamma
)