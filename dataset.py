import torch, json, glob
import numpy as np
import pandas as pd
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

########################################################

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer 
pretrained_weights = 'cl-tohoku/bert-base-japanese-whole-word-masking'

########################################################
##################  Cinnamon Dataset  ##################
class Cinnamon_Dataset(Dataset):
    def __init__(self, cinnamon_path, tokenizer):
        def get_tags(cinnamon_path):
            tags = set()
            files = glob.glob(f'{cinnamon_path}/ca_data/*')
            for file in files:
                dataframe = pd.read_excel(file, encoding="utf8")
                label_str = filter(lambda i:(type(i) is str), dataframe['Tag'])
                def split(strings):
                    out = list()
                    for string in strings: 
                        out += string.split(";")
                    return out
                items = split(label_str)
                tags.update(items)
            return tuple(sorted(list(tags)))
            #return tuple(["[PAD]", "[None]"] + sorted(list(tags)))
        
        def get_samples(cinnamon_path):
            groups = []
            files = glob.glob(f'{cinnamon_path}/ca_data/*')
            for file in files:
                dataframe = pd.read_excel(file, encoding="utf8")

                p_index = dataframe.groupby('Parent Index')
                for g in list(p_index.groups.keys()):
                    groups.append(p_index.get_group(g))
            return groups
        
        self.tokenizer = tokenizer
        self.samples = get_samples(cinnamon_path)
        self.tags = get_tags(cinnamon_path)

        print(f'\t[Info] Load Cannon_Dataset complete !! len:{self.__len__()}')    
        
    def __len__(self):
        return len(self.samples) 
    
    def __getitem__(self, idx):
        return self.samples[idx]
            
    def collate_fn(self, samples):        
        tokenizer, tags = self.tokenizer, self.tags
            
        CLS, SEP, PAD = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
        
        def zero_vec(): 
            return [0]*len(tags)
        
        ## text tokenized, label vectoized
        b_token_ids, b_output = [], []
        for sample in samples:
            token_ids = [CLS]
            output = [zero_vec()]
            for text, tag in zip(sample['Text'],sample['Tag']):
                ids = tokenizer.encode(text)[1:-1] + [SEP]
                label = zero_vec()
                if isinstance(tag, str): 
                    for t in tag.split(';'):
                        label[tags.index(t)] = 1
                token_ids += ids
                output += [label]*(len(ids)-1) + [zero_vec()]
            b_token_ids.append(token_ids)
            b_output.append(output)

        ## pad to same lenght
        max_len = min([max([len(s) for s in b_token_ids]), 512])
        for idx,(token_ids, output) in enumerate(zip(b_token_ids, b_output)):            
            token_ids = token_ids[:max_len]
            token_ids += [PAD]*(max_len-len(token_ids))
            b_token_ids[idx] = token_ids
            
            output = output[:max_len]
            output += [zero_vec()]*(max_len-len(output))
            b_output[idx] = output

        return torch.tensor(b_token_ids), torch.tensor(b_output)
    