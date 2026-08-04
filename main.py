import torch
from datasets.dataset import VerdictDataset
from datasets.preprocess import download_the_verdict
from datasets.dataloader import verdictDataLoader
from models.gpt2 import GPT

model_config = {
    "vocab_size" : 50257,
    "emb_dim" : 768,
    "n_layers" : 12,
    "n_heads" : 32,
    "dropout_rate": 0.2,
    "context_size" : 1024,
    "qkv-bias": False
    }

train_config = {
    "batch_size":4,
    "stride":5,
    "context_length":5,
    "shuffle":False
}

raw_text = download_the_verdict()
dataset = VerdictDataset(raw_text, train_config["context_length"], train_config['stride'])
dataloader = verdictDataLoader(dataset,train_config['batch_size'],shuffle=train_config['shuffle'])


dataiter = iter(dataloader)
sample_data = next(dataiter)


model = GPT(model_config)

def cross_entropy_loss(output, target):
    probas = output[:,:,target]
    loss = sum(torch.log(probas))
    return loss

output = model(sample_data[0])
print(output)


