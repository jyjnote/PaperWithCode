import torch


def load_data(data_path):
    import pandas as pd

    train = pd.read_csv(data_path + "\\review_train.csv")
    test = pd.read_csv(data_path + "\\review_test.csv")

    return train,test


import torch
class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, x, y=None):
        self.tokenizer = tokenizer
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        item = {}
        item["x"] = self.get_tokenizer(self.x[idx])
        
        if self.y is not None:
            item["y"] = torch.Tensor(self.y[idx])
        return item
    def get_tokenizer(self, text):
        x = self.tokenizer(text, padding="max_length", truncation=True)
        for k, v in x.items():
            x[k] = torch.tensor(v)
        return x

class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, x, y = None):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx] if self.y is not None else None
    

class CollateFN:
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, batch):
        import numpy as np
        x_list = []
        y_list = []
        for x, y in batch:
            x_list.append(x)
            if y is not None: # 정답데이터가 있을 경우
                y_list.append(y)

        x = self.tokenizer(x_list, padding=True, truncation=True, max_length=self.tokenizer.model_max_length,return_tensors="pt")

        y = torch.Tensor(np.array(y_list)) if len(y_list) else None
        return { "x":x, "y":y }
