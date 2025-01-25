import torch
import numpy as np
def train_loop(dataloader, model, loss_fn, optimizer, device):
    epoch_loss = 0
    model.train() # 학습 모드
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        pred = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).logits
        loss = loss_fn( pred, batch["y"].to(device) )

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
    act_func = torch.nn.Sigmoid()
    model.eval() # 평가 모드
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        pred = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).logits
        if batch.get("y") is not None:
            loss = loss_fn( pred, batch["y"].to(device) )
            epoch_loss += loss.item()

        pred = act_func(pred) # logit 값을 확률로 변환
        pred = pred.to("cpu").numpy() # cpu 이동후 ndarray 로변환
        pred_list.append(pred)

    epoch_loss /= len(dataloader)
    pred = np.concatenate(pred_list)
    return epoch_loss, pred


def start_training(train_input_ids,
                   train_token_type_ids,
                   train_attention_mask,
                   target,
                   device,
                   model_name= "bert-base-uncased"):
    from tqdm import tqdm
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score
    from transformers import AutoModelForSequenceClassification
    from data import ReviewDataset

    best_score_list = []
    n_splits = 5
    kf = KFold(n_splits, shuffle=True, random_state=42)

    batch_size = 16 # 배치 사이즈
    loss_fn = torch.nn.BCEWithLogitsLoss() # 손실 객체
    epochs = 100 # 최대 가능한 에폭수


    for i, (tri, vai) in enumerate( kf.split(train_input_ids) ):
    # 학습용 데이터로더 객체
        train_dt = ReviewDataset(train_input_ids[tri],train_token_type_ids[tri],train_attention_mask[tri], target[tri])
        train_dl = torch.utils.data.DataLoader(train_dt, batch_size=batch_size, shuffle=True)

    # 검증용 데이터로더 객체
        valid_dt = ReviewDataset(train_input_ids[vai],train_token_type_ids[vai],train_attention_mask[vai], target[vai])
        valid_dl = torch.utils.data.DataLoader(valid_dt, batch_size=batch_size, shuffle=False)

    # 모델 객체와 옵티마이저 객체 생성
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device)
        optimizer = torch.optim.Adam( model.parameters(), lr = 3e-5 )

        best_score = 0 # 현재 최고 점수
        patience = 0 # 조기 종료 조건을 주기 위한 변수
        for epoch in tqdm(range(epochs)):
            train_loss = train_loop(train_dl, model, loss_fn, optimizer, device)
            valid_loss, pred = test_loop(valid_dl, model, loss_fn, device)

            pred = (pred > 0.5).astype(int) # 이진분류 문제에서 클래스 번호 결정
            score = accuracy_score(target[vai], pred)

            patience += 1
        #print(train_loss, valid_loss, score)
            if score > best_score:
                best_score = score # 최고 점수 업데이트
                patience = 0
                #torch.save(model.state_dict(), f"{DATA_PATH}/weight/bert_model_{i}.pth") # 최고 점수 모델 가중치 저장


            if patience == 10:
                break

        print(f"{i}번째 폴드 최고 정확도: {best_score}")
        best_score_list.append(best_score)

        import gc
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    print("평균 점수:", sum(best_score_list)/len(best_score_list))
