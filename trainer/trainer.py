import math
from evaluation.loss import token_accuracy,calc_loss_accuracy_loader
from generation.sample_text import generate_sample_text
from utils.checkpoint import save_checkpoint
from pathlib import Path

def train_simple_model(epoch,model,device,tokenizer,metrics,dataloader,loss,optimizer,scheduler, val_dataloader):
    global_step = 0
    model.train()
    for i in range(epoch):
        epoch_loss = 0
        epoch_acc = 0
        step_size = 0
        for x,y in dataloader:
            x = x.to(device)
            y = y.to(device)
            print(f"step = {step_size}")
            optimizer.zero_grad()
            output = model(x)
            lossx = loss(output, y)
            lossx.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += lossx.detach().cpu().numpy()
            epoch_acc += token_accuracy(output,y)
            step_size +=1
            global_step +=1
        epoch_acc = epoch_acc / len(dataloader)
        epoch_loss = epoch_loss / len(dataloader)
        val_loss , val_acc = calc_loss_accuracy_loader(model,device,val_dataloader)
        print(f"after {i+1} epoch global step {global_step} the train loss {epoch_loss} val loss {val_loss} and train acc| {epoch_acc} val acc| {val_acc} ")
        print(f"text generate after epoch {i + 1} is {generate_sample_text(model,device ,tokenizer,'Every effort moves you',8)}")
        metrics.update(
                    train_loss=epoch_loss,
                    val_loss=val_loss,
                    train_acc=epoch_acc.cpu(),
                    val_acc=val_acc.cpu(),
                    train_ppl=math.exp(epoch_loss),
                    val_ppl=math.exp(val_loss) ,
        )
        save_checkpoint(Path("check_store"),model, optimizer, global_step, epoch, metrics)