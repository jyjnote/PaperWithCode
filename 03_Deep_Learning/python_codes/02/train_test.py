from dataset import TitanicDataset
from model import NN
from data_processing import load_and_process_data
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import torch
import numpy as np

train_ft, test_ft, target = load_and_process_data()


batch_size=11 # 6일때는 안됨
loss_fn=torch.nn.BCEWithLogitsLoss()
device = "cuda" if torch.cuda.is_available() else "cpu"
epochs=5
n_splits=5
n_features= train_ft.shape[1]


# model=NN(n_features).to(device)
# optimizer= torch.optim.Adam(model.parameters())

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

cv=KFold(n_splits=n_splits,shuffle=True,random_state=42)

def train_loop(dataloader,model,loss_fn,optimizer, device):
    epoch_loss=0
    model.train()

    for batch in dataloader:
        pred=model(batch['x'].to(device))
        loss=loss_fn(pred,batch['y'].to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss+=loss.item()
    epoch_loss/=len(dataloader)

    return epoch_loss

@torch.no_grad()
def test_loop(dataloader,model,loss_fn,device):
    model.eval()
    epoch_loss=0
    act_func=torch.nn.Sigmoid()
    pred_list=[]

    for batch in dataloader:
        pred=model(batch['x'].to(device))

        if batch.get('y') is True:
            loss=loss_fn(pred,batch['y'].to(device))
            epoch_loss+=loss.item()

        pred=act_func(pred).to('cpu').numpy()
        pred_list.append(pred)

    epoch_loss/=len(dataloader)
    pred=np.concatenate(pred_list)

    return epoch_loss,pred

best_score_list = []

for i, (train_idx, valid_idx) in enumerate(cv.split(train_ft)):
    # 데이터 분할
    x_train, y_train = train_ft[train_idx], target[train_idx]
    x_valid, y_valid = train_ft[valid_idx], target[valid_idx]

    # 데이터로더 생성
    train_dl = torch.utils.data.DataLoader(
        TitanicDataset(x_train, y_train), batch_size=batch_size, shuffle=True
    )

    valid_dl = torch.utils.data.DataLoader(
        TitanicDataset(x_valid, y_valid), batch_size=batch_size, shuffle=False
    )

    # 모델, 옵티마이저 설정
    model = NN(n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    
    best_score, patience = 0, 0
    
    for epoch in tqdm(range(epochs), desc=f"Fold {i+1}"):
        train_loss = train_loop(train_dl, model, loss_fn, optimizer, device)
        valid_loss, pred = test_loop(valid_dl, model, loss_fn, device)
        
        score = roc_auc_score(y_valid, pred)
        
        if score > best_score:
            best_score = score
            patience = 0
            #torch.save(model.state_dict(), os.path.join(save_dir, f"model_titanic_{i}.pth"))
        else:
            patience += 1
        
        if patience >= 20:
            break

    print(f"Fold {i+1} - Best AUC: {best_score:.4f}")
    best_score_list.append(best_score)
