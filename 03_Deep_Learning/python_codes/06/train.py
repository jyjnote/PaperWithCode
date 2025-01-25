import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
import random
import os

def reset_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train_loop(model,optimizer,dataloader,loss_fn,device):
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        preds=model(batch['image'].to(device))
        loss=loss_fn(preds, batch['target'].to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() 

    return running_loss / len(dataloader)
@torch.no_grad()
def test_loop(model,dataloader,loss_fn,device):
    model.eval()
    epoch_loss = 0.0
    preds_list=[]
    act_func = torch.nn.Sigmoid()

    for batch in dataloader:
        preds = model(batch['image'].to(device))
        
        if batch.get('y'):
            loss = loss_fn(preds, batch['target'].to(device))
            running_loss += loss.item()
        
        pred=act_func(preds)
        preds_list.append(pred.to('cpu').numpy())

    epoch_loss = epoch_loss / len(dataloader)
    preds = np.concatenate(preds_list)

    return epoch_loss, preds

n_split = 5
batch_size =20
epochs=1000
loss_fn=torch.nn.BCEWithLogitsLoss()

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

cv=KFold(n_splits=n_split,shuffle=True,random_state=42)

from glob import glob
import os
import numpy as np


# 고양이 이미지 파일 경로를 담은 리스트
cats_list = sorted(glob(os.path.join(r"C:\PapersWithCode\03_Deep_Learning\data\cats_and_dogs\train\cats", "*.jpg")))

# 개 이미지 파일 경로를 담은 리스트
dogs_list = sorted(glob(os.path.join(r"C:\PapersWithCode\03_Deep_Learning\data\cats_and_dogs\train\dogs", "*.jpg")))




labels = [0] * len(cats_list) + [1] * len(dogs_list) # 정답데이터 만들기
img_path = cats_list + dogs_list # 고양이와 개 이미지 파일 경로 합치기

# 멀티 인덱싱을 위해 ndarray 로 변환
train = np.array(img_path)
target = np.array(labels)

np.random.seed(42) # 동일한 shuffle 위해 시드 고정

# 인덱스를 이용하여 섞기 위해 샘플 개수 만큼 인덱스 생성
index_arr = np.arange(train.shape[0])

# 섞기

np.random.shuffle(index_arr)

# shuffle 된 인덱스를 이용하여 샘플 섞기
train = train[index_arr]
target = target[index_arr]
target = target.reshape(-1, 1)


# GPU 사용
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from data_processing import CatDogDataset,train_transform,test_transform
from model import Net

best_score_list=[]

for fold, (train_index, valid_index) in enumerate(cv.split(train)):
    x_train,y_train = train[train_index],target[train_index]
    x_valid,y_valid = train[valid_index],target[valid_index]

    # DataLoader로 DataLoader 만들기
    train_dataset = CatDogDataset(train_transform,x_train, y_train)
    valid_dataset = CatDogDataset(test_transform,x_valid, y_valid)

    train_dl=torch.utils.data.DataLoader(train_dataset,shuffle=True,batch_size=batch_size)
    valid_dl=torch.utils.data.DataLoader(valid_dataset,shuffle=False,batch_size=batch_size)

    model=Net()
    model.to(device)
    optimizer = torch.optim.Adam( model.parameters() )
    best_score = -np.inf # 현재 최고 점수
    patience = 0 # 조기 종료 조건을 주기 위한 변수

    for epoch in tqdm(range(epochs)):
        train_loss = train_loop(model, optimizer, train_dl, loss_fn, device)
        valid_loss, pred = test_loop(model, valid_dl, loss_fn, device)

        pred=(pred>0.5).astype(int)
        scores=accuracy_score(y_valid, pred)

        if scores > best_score:
            best_score = scores # 최고 ��수 업���이트
            patience = 0 # patience ���기화
        else:
            patience += 1 # patience ���가
            if patience > 10: # patience 10회 이상 ��어서면 early stopping
                break

    print(f"{fold}번째 폴드 최고 정확도: {best_score}")   
    best_score_list.append(best_score)

