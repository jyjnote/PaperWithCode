

def data_loda(data_path):
    import pandas as pd

    train,test=pd.read_csv(data_path+"\\review_train.csv"),pd.read_csv(data_path+"\\review_test.csv")
    target = train["target"].to_numpy().reshape(-1,1)

    return train, target, test

def data_processing(train, test):
    from kiwipiepy import Kiwi

    kiwi=Kiwi()

    gen=kiwi.tokenize(train["review"])
    train_data=[]
    test_data=[]
    for tokens in gen:
        tmp = [ t.form  for t in tokens ]
        train_data.append(tmp)

    for tokens in gen:
        tmp=[t.form for t in tokens]
        test_data.append(tmp)

    from torchtext.vocab import build_vocab_from_iterator
    vocab = build_vocab_from_iterator(train_data, specials=["<pad>","<unk>"])
    vocab.set_default_index( vocab["<unk>"] )

    train_data=[ vocab(text) for text in train_data ]
    test_data = [ vocab(text)  for text in test_data ]

    return train_data,test_data,vocab

import torch

class Dataset(torch.utils.data.Dataset):  # Use 'MyDataset' to avoid confusion with torch Dataset
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return {
            "data": torch.tensor(self.data[index]),
            "labels": torch.tensor(self.labels[index], dtype=torch.float32)
        }

def collate_fn(dataset_lst):
    src_lst = []
    trg_lst = []
    for dataset in dataset_lst:
        src_lst.append( dataset["data"] )
        trg_lst.append(dataset['labels'])

    src = torch.nn.utils.rnn.pad_sequence(src_lst, batch_first=True)
    trg = torch.stack(trg_lst)

    return {"src": src, "trg": trg}

from torch.nn.utils.rnn import pad_sequence
# def collate_fn(batch):
#    # batch는 [(data, label), (data, label), ...] 형태입니다.
#    data, labels = zip(*batch)
#    
#    # 시퀀스 데이터는 패딩을 사용하여 길이를 맞춰줍니다.
#    padded_data = pad_sequence(data, batch_first=True, padding_value=0)
#    
#    return padded_data, torch.stack(labels)


class VocabBuilder:
    def __init__(self, train_data):
        self.train_data = train_data
        self.vocab = self.build_vocab()

    def build_vocab(self):
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
        gen = kiwi.tokenize(self.train_data["review"])
        tokenized_data = [ [t.form for t in tokens] for tokens in gen ]
        
        from torchtext.vocab import build_vocab_from_iterator
        vocab = build_vocab_from_iterator(tokenized_data, specials=["<pad>", "<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        
        return vocab

    def get_vocab(self):
        return self.vocab