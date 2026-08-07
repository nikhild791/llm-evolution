import torch
from pathlib import Path

def save_checkpoint(path, model, optimizer, step, epoch, loss):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": model.config.to_dict(),
        "epoch": epoch,
        "step": step,
        "loss": loss,
    }

    temp_file = path / "checkpoint.tmp"
    final_file = path / f"checkpoint_{step}.pt"

    torch.save(checkpoint, temp_file)

    temp_file.replace(final_file)

    print(f"Checkpoint saved -> {final_file}")

