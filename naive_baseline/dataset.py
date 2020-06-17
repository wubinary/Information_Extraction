import json, glob, torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import re
import unicodedata

########################################################
##################  Examples Dataset  ##################
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
                    out = [unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', tag)) for tag in out]
                    return out
                items = split(label_str)
                tags.update(items)
            return tuple(sorted(list(tags)))
        
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
        
        def sub_idx_finder(list1, list2):            
            for i in range(len(list1)-len(list2)):
                find = True
                hit, miss = 0, 0
                for j in range(len(list2)):
                    if list1[i+j] != list2[j]: 
                        find = False
                        miss += 1
                    else:
                        hit += 1
                if miss < len(list2)/5:
                    find = True
                if find:
                    return i
            #print('yeh')
        
        ## text tokenized, label vectoized
        b_token_ids, b_output = [], []
        for sample in samples:
            token_ids = [CLS]
            output = [zero_vec()]
            for text, tag, value in zip(sample['Text'],sample['Tag'],sample['Value']):
                # 全形半形問題
                text = str(unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', text)))
                tag = str(unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', tag))) if tag is not np.nan else tag
                value = str(unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', value))) if value is not np.nan else value
                    
                ###
                ids = tokenizer.encode(text)[1:-1] + [SEP]
                labels = [zero_vec()]*(len(ids)-1) + [zero_vec()]
                
                if isinstance(tag, str):
                    for t,v in zip(tag.split(';'), str(value).split(';')):
                        t = unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', t))
                        v = unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', v))
                        
                        ids_v = tokenizer.encode(v)[1:-1]
                        pivote = sub_idx_finder(ids, ids_v)
                        for k in range(len(ids_v)):
                            if pivote is not None:
                                labels[pivote+k][tags.index(t)] = 1
                token_ids += ids
                output += labels
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