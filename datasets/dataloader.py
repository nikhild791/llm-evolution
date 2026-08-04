from torch.utils.data import DataLoader

def verdictDataLoader(dataset,batch_size, shuffle,drop_last=False,num_workers=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)