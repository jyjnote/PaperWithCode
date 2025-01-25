import torch
import numpy as np
def train_loop(model,loss_fn,dataloader,device,optimize):
    model.train()
    epoch_loss=0

    for batch in dataloader:
        preds=model(batch['x'].to(device))
        loss=loss_fn(preds, batch['y'].to(device))

        optimize.zero_grad()
        loss.backward()
        optimize.step()

    epoch_loss+=loss.item()

    return epoch_loss/len(dataloader)

@torch.no_grad()
def test_loop(model, loss_fn, dataloader, device):
    model.eval()
    epoch_loss=0
    act_func=torch.nn.Softmax(dim=1)
    pred_list=[]


    for batch in dataloader:
        
        x = batch['x'].to(device)
        preds = model(x)
            
        if batch.get('y') is not None:
            y = batch['y'].to(device)

            loss = loss_fn(preds, y)
            epoch_loss += loss.item()

        preds = act_func(preds) # logit 값을 확��로 변환
        preds = preds.to('cpu').numpy() # cpu 이������ ndarray 로변환
        pred_list.append(preds)

    epoch_loss /= len(dataloader)
    pred_list = np.concatenate(pred_list)

    return epoch_loss,pred_list

def start_training(model,kf_for_train,train_path,target,batch_size,device):
    from sklearn.model_selection import KFold
    from sklearn.metrics import  f1_score
    from tqdm import tqdm

    kf=KFold(n_splits=5,shuffle=True,random_state=42)
    loss_fn = torch.nn.CrossEntropyLoss()

    epochs=100

    from data_processing import MeatDataset,train_transform

    for i,(train_idx,val_idx) in enumerate(kf.split(kf_for_train)):
        
        #train_idx, val_idx = tri, vli
        #train_path, val_path = train_path[train_idx], train_path[val_idx]

        # MeatDataset을 생성
        #train_dataset = MeatDataset(train_idx, val_idx,train_transform)
        # DataLoader 생성
        #train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        #print(next(iter(train_dl)))
        train_paths, val_paths = train_path[train_idx], train_path[val_idx]
        train_targets, val_targets = target[train_idx], target[val_idx]

        # MeatDataset 생성
        train_dataset = MeatDataset(train_paths, train_targets, train_transform)
        val_dataset = MeatDataset(val_paths, val_targets, train_transform)

        # DataLoader 생성
        train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        model.to(device)
        optimizer = torch.optim.Adam( model.parameters() ) 


        
        for epoch in tqdm( range(epochs) ): # epochs 변수에 정의된 횟수 만큼의 반복문
            train_loss = train_loop(model,loss_fn,train_dl,  device,optimizer) # 학습하는 함수(학습 loss 반환)
            #valid_loss, pred = test_loop(valid_dl, model, loss_fn, device) # 검증loss 및 예측값을 반환 하는 함수
            #pred = np.argmax(pred, axis=1) # 각 예측값별로 가장 높은 확률의 열부분 인덱스(클래스 번호) 구함
            #score = f1_score(y_valid, pred, average="micro") # 현재 에폭의 검증 점수
        
            print(train_loss)

        