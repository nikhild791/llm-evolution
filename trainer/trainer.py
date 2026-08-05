import math
from evaluation.loss import token_accuracy,calc_loss_accuracy_loader
from generation.sample_text import generate_sample_text

def train_simple_model(epoch,model,device,metrics,tokenizer,dataloader,loss,optimizer, val_dataloader):
    for i in range(epoch):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        for x,y in dataloader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = model(x)
            lossx = loss(output, y)
            lossx.backward()
            optimizer.step()
            epoch_loss += lossx.detach().cpu().numpy()
            epoch_acc += token_accuracy(output,y)
        epoch_acc = epoch_acc / len(dataloader)
        epoch_loss = epoch_loss / len(dataloader)
        val_loss , val_acc = calc_loss_accuracy_loader(model,device,val_dataloader)
        print(f"after {i+1} epoch the train loss {epoch_loss} val loss {val_loss} and train acc| {epoch_acc} val acc| {val_acc} ")
        print(f"text generate after epoch {i + 1} is {generate_sample_text(model,device ,tokenizer,'Every effort moves you',8)}")
        metrics.update(
                    train_loss=epoch_loss,
                    val_loss=val_loss,
                    train_acc=epoch_acc.cpu(),
                    val_acc=val_acc.cpu(),
                    train_ppl=math.exp(epoch_loss),
                    val_ppl=math.exp(val_loss) ,
        )