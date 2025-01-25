
def read_data():
    import re
    with open(r"C:\PapersWithCode\04_NLP\data\generator\위대한유산.txt", encoding="utf-8") as f:
        text = f.read()

    pat = re.compile("[^a-zA-Z 가-힣ㄱ-ㅎ0-9.,\"\'\n!?]")
    text = pat.sub("", text)

    return text

def processing_data(text):
    chars = sorted(set(text))
    id2char = dict(enumerate(chars, 2))
    id2char[0] = "<pad>"
    id2char[1] = "<unk>"

    char2id = { v : k for k, v in id2char.items() }

    idx_list = [ char2id[c] for c in text]

    max_len = 61
    train = []
    for i in range(0, len(idx_list) + 1 - max_len, 3 ): # 윈도우 크기 이동
        train.append(idx_list[i:i+max_len] )
        #char_sequence = [id2char[idx] for idx in idx_list[i:i+max_len]]
    return id2char, char2id, train

import torch

class GenDataset(torch.utils.data.Dataset):
    def __init__(self, x):
        self.x = x
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx])
    
def collate_fn(lst):
    import random
    import numpy as np

    max_len = 30

    if random.random() < 0.5:
        max_len = np.random.randint(10,29)

    x = [tokens[:max_len] for tokens in lst]
    y = [tokens[max_len] for tokens in lst]

    return {"x":torch.stack(x) , "y":torch.stack(y) }