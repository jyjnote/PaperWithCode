import pandas as pd
import numpy as np

DATA_PATH=r'C:\PapersWithCode\04_NLP'

train = pd.read_csv(f"{DATA_PATH}/data/news/train_news.csv")
test = pd.read_csv(f"{DATA_PATH}/data/news/test_news.csv")
target = train["target"].to_numpy()
#print(train.head())

#vocab_title_data=train["title"]
vocab_desc_data=train["desc"]

from torchtext.data.utils import get_tokenizer
tokenizer = get_tokenizer("basic_english") 

import torchtext
torchtext.disable_torchtext_deprecation_warning()
def get_tokenizer(data, tokenizer):
    for text in data:
        yield tokenizer(text)

gen=get_tokenizer(vocab_desc_data, tokenizer)

from torchtext.vocab import build_vocab_from_iterator
vocab_desc=build_vocab_from_iterator(vocab_desc_data,specials=["<pad>","<unk>"])
vocab_desc.set_default_index(vocab_desc["<unk>"])

train_desc_data = [ vocab_desc( tokenizer(text) ) for text in train["desc"] ]
test_desc_data = [ vocab_desc( tokenizer(text) ) for text in test["desc"] ]

desc_max_len = max( len(lst) for lst in train_desc_data )



def pad(data,max_len):
    train_desc_data = []

    for text in data:
        tmp=text[:max_len]

        if len(tmp) < max_len:
            tmp =[0]*(max_len-len(text))+tmp

        train_desc_data.append(tmp)
    return np.array(train_desc_data)

train=pad(train_desc_data,desc_max_len)


import torch
class NewsDataset(torch.utils.data.Dataset):
    
    def __init__(self, desc_data, target):
        self.desc_data = desc_data
        self.target = target

    def __len__(self):
        return len(self.desc_data)
    
    def __getitem__(self, idx):
        return {'desc': torch.tensor(self.desc_data[idx]),
                'target': torch.tensor(self.target[idx]) if self.target is not None else None}