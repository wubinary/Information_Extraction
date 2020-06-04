import torch, json
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ABC_dataset(Dataset):
    def __init__(self):
        self.data = []

        print(f'\t[Info] Load ABC_Dataset complete !! len:{self.__len__()}')              
    def __len__(self):
        return len(self.data) 
    def __getitem__(self, idx):        
        context  = self.data['context'][idx]
    
def collate_fn(samples):

    return torch.tensor([])


