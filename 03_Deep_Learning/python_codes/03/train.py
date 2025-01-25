from data_processing import FinanceDataset,transform_data
from model import Resnet_Seqblock_LstmNet
import pandas_datareader.data as web
import torch
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error



df_1 = web.DataReader('005930', 'naver', start='2022-01-01', end='2022-12-31')
df_1 = df_1.astype(int)
data_1 = df_1.to_numpy()

train_x_arr,train_y_arr,mins,sizes=transform_data(data_1)

loss_fn=torch.nn.MSELoss()
device= "cuda" if torch.cuda.is_available() else "cpu"
n_splits = 5 # k-fold에 k값을 의미
input_size = train_x_arr.shape[2] # 입력 피처 개수
hidden_size = 16 # 순환신경망의 출력 피처 개수
batch_size = 32
epochs = 1000
loss_fn = torch.nn.MSELoss()


def train_loop(model,optimizer,loss_fn,data_loader,device):
    model.train()
    epoch_loss=0

    for batch in data_loader:
        preds = model(batch['x'].to(device))
        loss = loss_fn(preds,batch['y'].to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss+=loss.item() 

    return epoch_loss/len(data_loader)

@torch.no_grad()
def test_loop(model,loss_fn,data_loader,device):
    model.eval()
    epoch_loss=0
    preds_list=[]

    for batch in data_loader:
        preds = model(batch['x'].to(device))

        if batch['y'] is not None:
             loss = loss_fn(preds,batch['y'].to(device))
             epoch_loss += loss.item()

        preds =preds.to("cpu").numpy()
        preds_list.append(preds)

    epoch_loss /= len(data_loader)
    preds = np.concatenate(preds_list)
    return epoch_loss, preds


is_holdout = False
best_score_list = []

cv=KFold(n_splits=n_splits,shuffle=True,random_state=42)

from tqdm import tqdm

for i,(train_idx,valid_idx) in enumerate(cv.split(train_x_arr)):

    train_data=train_x_arr[train_idx]
    train_target=train_y_arr[train_idx]
    valid_data=train_x_arr[valid_idx]
    valid_target=train_y_arr[valid_idx]

    train_dataset=FinanceDataset(train_data,train_target)
    valid_dataset=FinanceDataset(valid_data,valid_target)

    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    valid_loader=torch.utils.data.DataLoader(valid_dataset,batch_size=batch_size,shuffle=False)
    
    model=Resnet_Seqblock_LstmNet(input_size,hidden_size).to(device)
    optimizer = torch.optim.Adam( model.parameters() )

    best_score = -np.inf # 현재 최고 점수
    patience = 0 # 조기 종료 조건을 주기 위한 변수

    for epoch in tqdm(range(epochs)):
        train_loss = train_loop(model, optimizer, loss_fn, train_loader, device)
        valid_loss, pred = test_loop(model, loss_fn, valid_loader, device)

        true = valid_target * sizes[3] + mins[3] # 원래에 수치로 복원
        pred = pred * sizes[3] + mins[3] # 원래에 수치로 복원
        score = mean_absolute_error(true, pred)

        if score > best_score:
            best_score = score # 최고 점수 업데이트
            patience = 0
        
        patience += 1
        if patience == 100:
            break
        
    print(f"{i}번째 폴드 BEST MAE: {best_score}")
    best_score_list.append(best_score)