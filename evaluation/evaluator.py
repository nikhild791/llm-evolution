import torch 
from .losses import CrossEntropyLoss, TokenAccuracy, Perplexity

class ValidationEngine:
    def __init__(self,loss, metrics):
        self.loss = loss
        self.metrics = metrics

    def vadidate(self, model,device ,loader):
        with torch.no_grad():
            model.eval()
            loader_loss= 0
            loader_acc = 0
            for x,y in loader:
                x = x.to(device)
                y = y.to(device)
                output = model(x)
                loss = CrossEntropyLoss(output, y)
                acc = TokenAccuracy(output,y)
                loader_loss += loss.detach().cpu().numpy()
                loader_acc += acc
            loader_loss /= len(loader)
            loader_acc /= len(loader)
            return{"loss": loader_loss.item() ,"accuracy":loader_acc.item(), "perplexity":torch.exp(loader_loss.item()) }
