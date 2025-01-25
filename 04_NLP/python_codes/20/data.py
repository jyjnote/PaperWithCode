
def data_load(data_path):
    import pandas as pd
    df=pd.read_csv(data_path)

    #data=df["review"].str.replace("[^a-zA-Z]", " ")
    data=df["review"]
    target = df["sentiment"].to_numpy().reshape(-1, 1)
    return data.tolist(),target

def data_processing(data,model_name = "bert-base-uncased"):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train = tokenizer(data, padding="max_length", truncation=True)

    return train

import torch

class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, token_type_ids, attention_mask, y=None):
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.y = y
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {}
        item["input_ids"] = torch.tensor( self.input_ids[idx] )
        item["token_type_ids"] = torch.tensor( self.token_type_ids[idx] )
        item["attention_mask"] = torch.tensor( self.attention_mask[idx] )

        if self.y is not None:
            item["y"] = torch.Tensor( self.y[idx] )
        return item
    