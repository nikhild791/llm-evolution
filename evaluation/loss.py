import torch

### calculate the cross entropy loss

# def cross_entropy_loss(output, target):
#     return F.cross_entropy(output.view(-1, output.size(-1)), target.view(-1))

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

## calculating loss and accuracy on val dataset
def calc_loss_accuracy_loader(model,device,dataloader):
    with torch.no_grad():
        model.eval()
        loader_loss= 0
        loader_acc = 0
        for x,y in dataloader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            loss = cross_entropy_loss(output, y)
            acc = token_accuracy(output,y)
            loader_loss += loss.detach().cpu().numpy()
            loader_acc += acc
        loader_loss /= len(dataloader)
        loader_acc /= len(dataloader)
        return loader_loss.item() ,loader_acc.item()
