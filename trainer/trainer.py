from evaluation.loss import token_accuracy,calc_loss_accuracy_loader

def train_simple_model(epoch,model,metrics,dataloader,loss,optimizer,val_dataloader):
    for _ in range(epoch):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        for x,y in dataloader:
            optimizer.zero_grad()
            output = model(x)
            lossx = loss(output, y)
            lossx.backward()
            optimizer.step()
            epoch_loss += lossx.detach().numpy()
            epoch_acc += token_accuracy(output,y)
        epoch_acc = epoch_acc / len(dataloader)
        epoch_loss = epoch_loss / len(dataloader)
        val_loss , val_acc = calc_loss_accuracy_loader(model,val_dataloader)
        metrics.update(
                    train_loss=epoch_loss,
                    val_loss=val_loss,
                    train_acc=epoch_acc,
                    val_acc=val_acc,
                    train_ppl=math.exp(epoch_loss),
                    val_ppl=math.exp(val_loss) ,
        )