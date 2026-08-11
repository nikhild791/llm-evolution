#? code to download the  the verdict story from sebestian rascheca
import os
import math
import numpy as np


def prepare_dataset(raw_text,tokenizer,training_config,dir_name):
    data_dir = os.path.join(os.getcwd() , 'data',dir_name)
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        tokenized_text = tokenizer.encode(raw_text)

        train_data = training_config.train_data_ratio  ## train data to be 90%
        len_train_data = math.floor(len(tokenized_text)* train_data)
        train_ids = tokenized_text[:len_train_data]
        val_ids = tokenized_text[len_train_data:]
        print(f"train has {len(train_ids):,} tokens")
        print(f"val has {len(val_ids):,} tokens")    
        save_train_val_bin(train_ids,val_ids,dir_name)
        return train_ids, val_ids
    train_ids = np.memmap(os.path.join(data_dir,'train.bin'), dtype=np.uint16, mode='r')
    val_ids = np.memmap(os.path.join(data_dir,'val.bin'), dtype=np.uint16, mode='r')
    return train_ids,val_ids

def save_train_val_bin(train_ids,val_ids ,dir_name):
    data_dir = os.path.join(os.getcwd() , 'data',dir_name)
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")

    if not os.path.exists(train_path):
        train_ids = np.array(train_ids, dtype=np.uint16)
        train_ids.tofile(os.path.join(data_dir, 'train.bin'))
    if not os.path.exists(val_path):
        val_ids = np.array(val_ids, dtype=np.uint16)
        val_ids.tofile(os.path.join(data_dir, 'val.bin'))

