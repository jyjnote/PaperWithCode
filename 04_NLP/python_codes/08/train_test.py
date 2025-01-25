import torch
import numpy as np
def train(model,device,dataloader,loss_fn,optimizer):
    model.train()
    epochs_loss = 0
    
    for batch in dataloader:
        preds=model(batch['data'].to(device))

        loss = loss_fn(batch['labels'].to(device,),preds)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epochs_loss += loss.item()
    
    return epochs_loss / len(dataloader)

@torch.no_grad()
def test(model,device,dataloader,loss_fn):
    model.eval()
    epochs_loss = 0
    preds_list=[]
    act_func=torch.nn.Softmax(dim=1)

    for batch in dataloader:
        preds=model(batch['data'].to(device))
        

        if batch.get("labels"):
            loss = loss_fn(preds,batch['labels'].to(device))
            epoch_loss += loss.item()
        
        preds=act_func(preds).to('cpu').numpy()
        preds_list.append(preds)

    preds=np.concatenate(preds_list)
    
    return epochs_loss / len(dataloader),preds

def train_test(vocab_size,train_data,target):
    from sklearn.model_selection import KFold
    from sklearn.metrics import f1_score
    from data_processing import NewsDataset
    from model import Net
    from tqdm import tqdm



    n_splits = 5
    cv = KFold(n_splits, shuffle=True, random_state=42)

    batch_size = 32 # 배치 사이즈
    loss_fn = torch.nn.CrossEntropyLoss()
    epochs = 100 # 최대 가능한 에폭수
    device="cuda" if torch.cuda.is_available() else "cpu"
    embedding_dim = 64 # 임베딩 벡터 크기
    best_score_list = []

    for i, (tri, vai) in enumerate( cv.split(train_data) ):
        # 학습용 데이터로더 객체
        train_dt = NewsDataset(train_data[tri], target[tri])
        train_dl = torch.utils.data.DataLoader(train_dt, batch_size=batch_size, shuffle=True)


        #print(next(iter(train_dl)))

        #break

        # 검증용 데이터로더 객체
        valid_dt = NewsDataset(train_data[vai], target[vai])
        valid_dl = torch.utils.data.DataLoader(valid_dt, batch_size=batch_size, shuffle=False)

        # 모델 객체와 옵티마이저 객체 생성
        model = Net(vocab_size,embedding_dim,4).to(device)
        optimizer = torch.optim.Adam( model.parameters() )

        best_score = 0 # 현재 최고 점수
        patience = 0 # 조기 종료 조건을 주기 위한 변수
        for epoch in tqdm(range(epochs)):
            train_loss = train(model, device, train_dl, loss_fn, optimizer)
            valid_loss, pred = test(model, device, valid_dl, loss_fn)

            pred = np.argmax(pred, axis=1) # 다중분류 문제에서 클래스 번호 결정
            score = f1_score(target[vai], pred, average="micro")

            #print(train_loss, valid_loss, score)
        
            if score > best_score:
                best_score = score # 최고 점수 업데이트
                patience = 0
                #torch.save(model.state_dict(), f"Conv1d_model_{i}.pth") # 최고 점수 모델 가중치 저장

            patience += 1
            if patience == 5:
                break

        print(f"{i}번째 폴드 최고 F1-Score micro: {best_score}")
        best_score_list.append(best_score)