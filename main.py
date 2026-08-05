import tiktoken
from datasets.dataset import VerdictDataset
from datasets.preprocess import download_the_verdict
from datasets.dataloader import verdictDataLoader
from evaluation.loss import cross_entropy_loss
from models.gpt2 import GPT
from configs.model import ModelConfig
from configs.training import TrainingConfig

GPT2_SMALL = ModelConfig(
    emb_dim=768,
    n_layers=12,
    n_heads=12,
    activation="gelu",
    context_length=8
)

TRAIN_CONFIG = TrainingConfig(
    epochs=1,
    batch_size=4,
    stride=8,
    context_length=8
)



raw_text = download_the_verdict()
dataset = VerdictDataset(raw_text, GPT2_SMALL.context_length, TRAIN_CONFIG.stride)
dataloader = verdictDataLoader(dataset,TRAIN_CONFIG.batch_size,shuffle=TRAIN_CONFIG.shuffle)


dataiter = iter(dataloader)
sample_data = next(dataiter)


model = GPT(GPT2_SMALL)
### goal to generate new tokens



# output = model(sample_data[0])

# print(output.shape)
# print(cross_entropy_loss(output, sample_data[1]))


