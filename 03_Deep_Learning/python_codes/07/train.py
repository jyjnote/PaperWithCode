import pandas as pd
import numpy as np
import torch
import torch.utils.data.dataloader
from tqdm.auto import tqdm



DATA_PATH = r'C:\\PapersWithCode\\03_Deep_Learning\\'
SEED = 42

from glob import glob

lst = glob(DATA_PATH+"/data/cats_and_dogs/train/cats/*.jpg") # 고양이 이미지 파일 경로를 담은 리스트
cats_list = sorted(lst, key = lambda x: x) # 파일 경로명으로 정렬

lst = glob(DATA_PATH+"/data/cats_and_dogs/train/dogs/*.jpg") # 고양이 이미지 파일 경로를 담은 리스트
dogs_list = sorted(lst, key = lambda x: x) # 파일 경로명으로 정렬

labels = [0] * len(cats_list) + [1] * len(dogs_list) # 정답데이터 만들기
img_path = cats_list + dogs_list # 고양이와 개 이미지 파일 경로 합치기

# 멀티 인덱싱을 위해 ndarray 로 변환
train = np.array(img_path)
target = np.array(labels)

np.random.seed(SEED) # 동일한 shuffle 위해 시드 고정

# 인덱스를 이용하여 섞기 위해 샘플 개수 만큼 인덱스 생성
index_arr = np.arange(train.shape[0])

# 섞기
np.random.shuffle(index_arr)
np.random.shuffle(index_arr)

# shuffle 된 인덱스를 이용하여 샘플 섞기
train = train[index_arr]
target = target[index_arr]

from data_processing import CatDogDataset,train_transform,test_transform

batch_size=32
target = target.reshape(-1,1)


# def train_loop(model,dataLoader,loss_fn,device,optimizer):
#     model.train()
#     total_loss = 0

#     for batch in dataLoader:
        
#         preds=model(batch['x'].to(device))
#         loss=loss_fn(preds,batch['y'].to(device))

#         optimizer.zero_grad(set_to_none=True) 
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     return total_loss / len(dataLoader)

def train_loop(dataloader, model, loss_fn, optimizer, device):
    epoch_loss = 0
    model.train() # 학습 모드
    for batch in dataloader:
        pred = model( batch["x"].to(device) )
        loss = loss_fn( pred, batch["y"].to(device) )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    epoch_loss /= len(dataloader)
    return epoch_loss

# @torch.no_grad()
# def test_loop(model,dataLoader,loss_fn,device):
#     model.eval()
#     total_loss = 0
#     preds_list=[]
#     act_func=torch.nn.Sigmoid()

#     for batch in dataLoader:
#         preds = model(batch['x'].to(device))
        
#         if batch['y'] is not None:
#             loss = loss_fn(preds, batch['y'].to(device))
#             total_loss += loss.item()

#         preds=act_func(preds)
#         preds=preds.to('cpu').numpy()
#         preds_list.append(preds)

#         total_loss/=len(dataLoader)
#         preds = np.concatenate(preds_list)
#         return total_loss, preds

@torch.no_grad()
def test_loop(dataloader, model, loss_fn, device):
    epoch_loss = 0
    pred_list = []
    act_func = torch.nn.Sigmoid()
    model.eval() # 평가 모드

    for batch in dataloader:
        pred = model( batch["x"].to(device) )

        if batch.get("y") is not None:
            loss = loss_fn( pred, batch["y"].to(device) )
            epoch_loss += loss.item()

        pred = act_func(pred) # logit 값을 확률로 변환
        pred = pred.to("cpu").numpy() # cpu 이동후 ndarray 로변환
        pred_list.append(pred)
        
    epoch_loss /= len(dataloader)
    pred = np.concatenate(pred_list)
    return epoch_loss, pred

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


kf = KFold(n_splits=5, shuffle=True, random_state=42)
epochs=100
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn=torch.nn.BCEWithLogitsLoss()
from tqdm import tqdm
from model import FreezNet2

best_score_list = []
for i, (tri, vai) in enumerate( kf.split(train) ):
    # 학습용 데이터
    x_train = train[tri]
    y_train = target[tri]

    # 검증용 데이터
    x_valid = train[vai]
    y_valid = target[vai]

    # 학습용 데이터로더 객체
    train_dt = CatDogDataset(train_transform, x_train, y_train)
    train_dl = torch.utils.data.DataLoader(train_dt, batch_size=batch_size, shuffle=True)

    # 검증용 데이터로더 객체
    valid_dt = CatDogDataset(test_transform, x_valid, y_valid)
    valid_dl = torch.utils.data.DataLoader(valid_dt, batch_size=batch_size, shuffle=False)

    # 모델 객체와 옵티마이저 객체 생성
    model = FreezNet2().to(device)
    optimizer = torch.optim.Adam( model.parameters() )
    best_score = 0 # 현재 최고 점수
    patience = 0 # 조기 종료 조건을 주기 위한 변수
    for epoch in tqdm( range(epochs) ):
        train_loss = train_loop(train_dl, model, loss_fn, optimizer, device)
        valid_loss, pred = test_loop(valid_dl, model, loss_fn, device)
        pred = (pred > 0.5).astype(int) # 확률 -> 클래스 값
        score = accuracy_score(y_valid, pred)

        #print(train_loss, valid_loss, score)
        
        if score > best_score:
            best_score = score # 최고 점수 업데이트
            patience = 0
            #torch.save(model.state_dict(), os.path.join(DATA_PATH,'weight','cats_dogs',f"model_{i}(resnet).pth")) # 최고 점수 모델 가중치 저장

        patience += 1
        if patience == 50:
            break
    print(f"{i}번째 폴드 최고 정확도: {best_score}")
    best_score_list.append(best_score)

# test=CatDogDataset(train, target,train_transform)
# dl=torch.utils.data.DataLoader(test,batch_size)
# print(next(iter(dl)))