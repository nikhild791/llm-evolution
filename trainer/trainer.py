import math
import torch
from dataclasses import asdict
from evaluation.losses import token_accuracy,calc_loss_accuracy_loader
from generation.sample_text import generate_sample_text,generate,token_ids_to_text,text_to_token_ids
from trainer.checkpoint import save_checkpoint
from pathlib import Path
from evaluation.metrics import Metrics
from tqdm.auto import tqdm
from configs.model import ModelConfig
from .scheduler import build_scheduler
from .optimizer import build_optimizer

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

## training infra v2
class Trainer:
    def __init__(self,model,tokenizer,train_dataloader,val_dataloader,device,loss, accuracy,checkpointconfig,training_config,optimizer_config,scheduler_config):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = build_optimizer(model, optimizer_config)
        self.scheduler = build_scheduler(self.optimizer, scheduler_config)
        self.epoch = training_config.epoch
        self.device = device
        self.loss = loss
        self.accuracy = accuracy
        self.metrics = Metrics()
        self.global_step = 0
        self.epoch_step = 0
        self.best_val_loss = 99999
        self.checkpointconfig = checkpointconfig

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

    def _generate_sample_text(self):
        token_ids = generate(
                     model=self.model,
                     idx=text_to_token_ids("Every effort moves you", self.tokenizer),
                     max_new_tokens=15,
                     context_size=self.model.config.context_length,
                     temperature=1
                    )
        return token_ids_to_text(token_ids, self.tokenizer)
    
    def _save_checkpoint(self, BEST=False):
        if self.checkpointconfig.path is not None:
            path = Path(self.checkpointconfig.path)
            path.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": asdict(self.model.config),
            "scheduler":self.scheduler.state_dict(),
            "epoch_step": self.epoch_step,
            "global_step": self.global_step,
            "metrics": dict(self.metrics.metrics),
            "best_val_loss": self.best_val_loss
            }
            temp_file = path / "checkpoint.tmp"
            final_file = path / f"checkpoint_{self.global_step}.pt"
            
            ### save best file
            if BEST:
                final_file =path / f"best_checkpoint.pt"
                torch.save(checkpoint, temp_file)
                temp_file.replace(final_file) 
                print(f"Best checkpoint saved -> {final_file}")
                ### remove the previous best checkpoint
                for file in Path.joinpath(Path.cwd(),path).glob("best*"):
                    if final_file.name != file.name:
                        print(final_file,"removed")
                        file.unlink() 
            else:
                    
                torch.save(checkpoint, temp_file)
                temp_file.replace(final_file)   
                print(f"Checkpoint saved -> {final_file}")
                ### keeping only last n files
                file_list = []
                if self.checkpointconfig.keep_last_n:
                    for file in Path.joinpath(Path.cwd(),path).glob("checkpoint*"):
                        file_list.append(file)
                    if len(file_list) > self.checkpointconfig.keep_last_n:
                        files = sorted(file_list,key=lambda  p: int(p.stem.rsplit("_", 1)[1]))[:-1*self.checkpointconfig.keep_last_n]
                        [file.unlink() for file in files]               

    def _resume_checkpoint(self, checkpoint_name):
        path = Path(self.checkpointconfig.path)
        checkpoint_path  = Path.joinpath(Path.cwd(),path , checkpoint_name)
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        self.model.config = ModelConfig(**checkpoint["config"])
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch_step = checkpoint['epoch_step']
        self.global_step = checkpoint['global_step']
        self.metrics.metrics.update(checkpoint['metrics'])
        self.best_val_loss = checkpoint["best_val_loss"]
    
    def _train_epoch(self):
        epoch_loss,epoch_acc = 0,0
        for batch in tqdm(self.train_dataloader):
            step_loss, step_acc = self._train_step(batch)
            epoch_loss +=step_loss
            epoch_acc += step_acc
            self.global_step +=1
        epoch_ppl = math.exp(epoch_loss/len(self.train_dataloader))
        return epoch_loss/len(self.train_dataloader), epoch_acc/len(self.train_dataloader), epoch_ppl

    def fit(self, resume_latest=False, resume_best=False):
        if resume_latest:
            file_list=[]
            for file in Path.joinpath(Path.cwd(),self.checkpointconfig.path).glob("checkpoint*"):
                        file_list.append(file)
            latest_checkpoint_file = sorted(file_list,key=lambda  p: int(p.stem.rsplit("_", 1)[1]))[-1]
            self._resume_checkpoint(latest_checkpoint_file.name)
        if resume_best:
            self._resume_checkpoint('best_checkpoint.pt')
        for i in range(self.epoch_step,self.epoch):
            self.model.train()
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
            ### periodic saving
            if self.global_step % self.checkpointconfig.save_every_steps == 0:
                self._save_checkpoint()
            ### best saving
            if val_loss < self.best_val_loss :
                self.best_val_loss = val_loss
                self._save_checkpoint(BEST=self.checkpointconfig.save_best)
            print("Output text:\n", self._generate_sample_text())
            print(f"after {i+1} epoch global step {self.global_step} the train loss {epoch_loss} val loss {val_loss} and train acc| {epoch_acc} val acc| {val_acc} ")
