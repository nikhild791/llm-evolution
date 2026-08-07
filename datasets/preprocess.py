#? code to download the  the verdict story from sebestian rascheca
import os
import math
import urllib.request 
from .dataloader import verdictDataLoader

def download_the_verdict():
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt"
    data_dir = os.path.join(os.getcwd() , 'data','the-verdict')
    os.makedirs(data_dir ,exist_ok=True)
    file_path = os.path.join(data_dir, 'the-verdict.txt')
    if not os.path.exists(file_path):
        urllib.request.urlretrieve(url,file_path)
    with open(file_path,'r') as f:
        return f.read()

def train_val_dataloader(raw_text,training_config):
    train_data = training_config.train_data_ratio  ## train data to be 90%
    len_train_data = math.floor(len(raw_text)* train_data)
    train_data = raw_text[:len_train_data]
    val_data = raw_text[len_train_data:] 


    train_dataloader = verdictDataLoader(train_data, training_config)
    val_dataloader = verdictDataLoader(val_data,training_config)

    return train_dataloader, val_dataloader