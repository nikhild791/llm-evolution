import torch

### calculate the cross entropy loss

def cross_entropy_loss(logits, targets):
    logits = logits.flatten(0,1)
    targets = targets.flatten()

    log_probs = torch.log_softmax(logits, dim=-1)

    loss = -log_probs[
        torch.arange(logits.size(0)),
        targets
    ].mean()

    return loss

### calculate the token accuracy per batch
def token_accuracy(logits, targets):
    """
    logits:  (B, T, V)
    targets: (B, T)
    """
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean()
    