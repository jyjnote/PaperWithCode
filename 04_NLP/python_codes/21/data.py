
def data_load(data_path):
    import pandas as pd
    df = pd.read_csv(data_path)
    return df

import torch
class ChatDataset(torch.utils.data.Dataset):
    def __init__(self,df):
        self.question = df["question"].tolist()
        self.answer = df["answer"].tolist()

    def __len__(self):
        return len(self.question)
    def __getitem__(self, idx):
        return "<q>" + self.question[idx] + "</s><a>" + self.answer[idx] + "</s>"
    

class CollateFN:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, batch):
        x = self.tokenizer(batch, return_tensors="pt",padding=True)
        return {"x" : x}
    
def train_loop(dataloader, model, loss_fn, optimizer, device):
    from tqdm import tqdm
    epoch_loss = 0
    model.train()
    for batch in tqdm(dataloader):
        x = batch["x"].to(device)
        pred = model(**x).logits # 예측값 batch, seq, n_class
        n_class = pred.shape[-1] # 정답 클래스 개수
        pred = pred[:,:-1].reshape(-1, n_class) # batch x seq, n_class

        trg = x["input_ids"][:,1:].flatten() # batch x seq

        mask = trg != 3
        trg = trg[mask]
        pred = pred[mask]

        loss = loss_fn(pred, trg)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(dataloader)

    return epoch_loss