import torch
import numpy as np
from tqdm import tqdm

SEED=42
def train_loop(dataloader, model, loss_fn, optimizer, device):
    epoch_loss = 0
    model.train() # 학습 모드
    
    for batch in dataloader:
        pred = model(batch["desc"].to(device) )
        loss = loss_fn( pred, batch["target"].to(device) )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(dataloader)
    return epoch_loss


@torch.no_grad()
def test_loop(dataloader, model, loss_fn, device):
    epoch_loss = 0
    pred_list = []
    act_func = torch.nn.Softmax(dim=1)
    model.eval() # 평가 모드
    
    for batch in dataloader:
        pred = model(batch["desc"].to(device))
        
        #print(pred)

        if batch.get("target") is not None:
            loss = loss_fn( pred, batch["target"].to(device) )
            epoch_loss += loss.item()

        pred = act_func(pred) # logit 값을 확률로 변환
        pred = pred.to("cpu").numpy() # cpu 이동후 ndarray 로변환
        pred_list.append(pred)

    epoch_loss /= len(dataloader)
    pred = np.concatenate(pred_list)
    return epoch_loss, pred

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
n_splits = 5
cv = StratifiedKFold(n_splits, shuffle=True, random_state=SEED)

batch_size = 32 # 배치 사이즈
loss_fn = torch.nn.CrossEntropyLoss() # 손실 객체
epochs = 100 # 최대 가능한 에폭수
device ="cuda" if torch.cuda.is_available() else "cpu"
embedding_dim = 64 # 임베딩 벡터 크기
hinnden_dim = 128
out_dim =4
def sart_training():
    from data_processing import NewsDataset,train,target,vocab_desc
    from model import Net
    # 모델 학습 및 저장 코드

    vocab_size=len(vocab_desc)
    best_score_list=[]

    for i, (tri, vai) in enumerate(cv.split(train, target)):
        # 학습용 데이터로더 객체
        train_dt = NewsDataset(train[tri],target[tri])
        train_dl = torch.utils.data.DataLoader(train_dt, batch_size=batch_size, shuffle=True)

        # 검증용 데이터로더 객체
        valid_dt = NewsDataset(train[vai],target[vai])
        valid_dl = torch.utils.data.DataLoader(valid_dt, batch_size=batch_size, shuffle=False)

        # 모델 객체와 옵티마이저 객체 생성
        model = Net(vocab_size, embedding_dim,hinnden_dim,out_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters())

        best_score = 0  # 현재 최고 점수
        patience = 0  # 조기 종료 조건을 주기 위한 변수

        for epoch in tqdm(range(epochs)):
            train_loss = train_loop(train_dl, model, loss_fn, optimizer, device)
            valid_loss, pred = test_loop(valid_dl, model, loss_fn, device)

            pred = np.argmax(pred, axis=1)  
            # pred의 예측값은 Softmax() 함수의 의해 확률값임
            # 다중분류 문제에서 클래스 번호 결정
            score = f1_score(target[vai], pred, average="micro")

            # 최고 점수 모델 가중치 저장
            if score > best_score:
                best_score = score  # 최고 점수 업데이트
                patience = 0
                #torch.save(model.state_dict(), os.path.join(DATA_PATH, 'weight', f"model_{i}.pth"))

            patience += 1
            if patience == 5:
                break

        print(f"{i}번째 폴드 최고 F1-Score micro: {best_score}")
        best_score_list.append(best_score)