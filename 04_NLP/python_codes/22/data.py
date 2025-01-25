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