import tiktoken
import torch

from configs.model import ModelConfig
from configs.training import TrainingConfig
from datasets.dataloader import verdictDataLoader
from datasets.preprocess import download_the_verdict
from evaluation.loss import cross_entropy_loss
from models.gpt2 import GPT

GPT2_SMALL = ModelConfig(
    emb_dim=32,
    n_layers=2,
    n_heads=4,
    activation="gelu",
    context_length=8,
)

TRAIN_CONFIG = TrainingConfig(
    epochs=1,
    batch_size=2,
    stride=2,
    context_length=8,
)

raw_text = download_the_verdict()

tokenizer = tiktoken.get_encoding("gpt2")
dataloader = verdictDataLoader(
    raw_text,
    batch_size=TRAIN_CONFIG.batch_size,
    shuffle=TRAIN_CONFIG.shuffle,
    context_length=TRAIN_CONFIG.context_length,
    stride=TRAIN_CONFIG.stride,
    tokenizer=tokenizer,
)

sample_batch = next(iter(dataloader))
model = GPT(GPT2_SMALL)

with torch.no_grad():
    logits = model(sample_batch[0])

print(logits.shape)
print(cross_entropy_loss(logits, sample_batch[1]))


