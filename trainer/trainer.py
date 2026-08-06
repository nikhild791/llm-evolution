import math
import torch
from dataclasses import asdict
from evaluation.loss import token_accuracy,calc_loss_accuracy_loader
from generation.sample_text import generate_sample_text
from utils.checkpoint import save_checkpoint
from pathlib import Path
from evaluation.metrics import Metrics


### v1 of writing training loop below and this is same but this is not modular the trainer is orchestrator but in this
### function trainer is doing every thing
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


class Trainer:
    def __init__(self,model,train_dataloader,val_dataloader,epoch,device,loss, accuracy, optimizer,scheduler,path=None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch = epoch
        self.device = device
        self.loss = loss
        self.accuracy = accuracy
        self.metrics = Metrics()
        self.global_step = 0
        self.epoch_step = 0
        self.path = path

    def _train_step(self,batch):
        x,y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(x)
        step_loss = self.loss(output, y)
        step_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        step_acc = self.accuracy(output, y)
        return step_loss.item(), step_acc.item()

    def _validate(self):
        val_loss , val_acc = calc_loss_accuracy_loader(self.model,self.device,self.val_dataloader)
        val_ppl = math.exp(val_loss)
        return val_loss, val_acc, val_ppl

    def _save_checkpoint(self):
        if self.path is not None:
            path = Path(self.path)
            path.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": asdict(self.model.config),
            "scheduler":self.scheduler.state_dict(),
            "epoch_step": self.epoch_step,
            "step": self.global_step,
            "metrics": self.metrics,
            }
            
            temp_file = path / "checkpoint.tmp"
            final_file = path / f"checkpoint_{self.global_step}.pt"
            
            torch.save(checkpoint, temp_file)
            
            temp_file.replace(final_file)
            
            print(f"Checkpoint saved -> {final_file}")
    
    def _train_epoch(self):
        epoch_loss,epoch_acc = 0,0
        for batch in self.train_dataloader:
            step_loss, step_acc = self._train_step(batch)
            epoch_loss +=step_loss
            epoch_acc += step_acc
            self.global_step +=1
        epoch_ppl = math.exp(epoch_loss/len(self.train_dataloader))
        return epoch_loss/len(self.train_dataloader), epoch_acc/len(self.train_dataloader), epoch_ppl

    def fit(self):
        self.model.train()
        for i in range(self.epoch):
            epoch_loss, epoch_acc, epoch_ppl = self._train_epoch()
            val_loss, val_acc, val_ppl = self._validate()
            self.metrics.update(
                                train_loss = epoch_loss,
                                train_acc = epoch_acc,
                                train_ppl = epoch_ppl,
                                val_loss = val_loss,
                                val_acc = val_acc,
                                val_ppl = val_ppl
                                )
            self.epoch_step +=1
            self._save_checkpoint()
            print(f"after {i+1} epoch global step {self.global_step} the train loss {epoch_loss} val loss {val_loss} and train acc| {epoch_acc} val acc| {val_acc} ")
