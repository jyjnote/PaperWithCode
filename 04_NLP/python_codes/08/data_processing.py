import numpy as np
import torch
def tokensize(train ):
    from torchtext.data.utils import get_tokenizer
    from torchtext.vocab import build_vocab_from_iterator
    import torchtext
    
    torchtext.disable_torchtext_deprecation_warning()

    tokenizer=get_tokenizer("basic_english")

    train_list=[tokenizer(text) for text in train]

    vocab=build_vocab_from_iterator(train_list,specials=["<pad>","<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    train_data = [ vocab(lst) for lst in train_list ]

    max_len = max( len(lst) for lst in train_data )

    train_data = [ [0] * ( max_len - len(lst) ) + lst  if len(lst) < max_len else lst[:max_len]  for lst in train_data  ]
    
    return len(vocab), np.array(train_data)


class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return {"data":torch.tensor(self.data[index]), 
                "labels":torch.tensor(self.labels[index]) if self.labels is not None else None}