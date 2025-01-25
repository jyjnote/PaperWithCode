import torch
import numpy as np
def train_loop(dataloader, model, loss_fn, optimizer, device):
    epoch_loss = 0
    model.train() # 학습 모드
    
    for batch in dataloader:
        pred = model( batch["src"].to(device) )
        loss = loss_fn( pred, batch["trg"].to(device) )

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
        pred = model( batch["src"].to(device) )
        
        if batch.get("trg") is not None:
            loss = loss_fn( pred, batch["trg"].to(device) )
            epoch_loss += loss.item()

        pred = act_func(pred) # logit 값을 확률로 변환
        pred = pred.to("cpu").numpy() # cpu 이동후 ndarray 로변환
        pred_list.append(pred)

    epoch_loss /= len(dataloader)
    pred = np.concatenate(pred_list)
    return epoch_loss, pred

from typing import Callable, List, Any
from sklearn.metrics import accuracy_score
import torch
from tqdm import tqdm
from sklearn.model_selection import KFold

def start_training(
    kf: KFold,  # KFold 교차 검증 객체
    train_data: List[Any],  # 훈련 데이터, 일반적으로 리스트 형태
    train,
    target: List[int],  # 이진 타겟 값 (0 또는 1)
    #dataset: torch.utils.data.Dataset,  # PyTorch Dataset 클래스
    batch_size: int,  # 배치 크기
    #Net: Callable[..., torch.nn.Module],  # 모델 클래스 (가변 인수로 생성 가능한 클래스)
    hp: dict,  # 하이퍼파라미터 딕셔너리
    device: str,  # "cpu" 또는 "cuda" (GPU)
    epochs: int,  # 훈련할 에폭 수
    loss_fn: torch.nn.Module,  # 손실 함수 (예: CrossEntropyLoss)
    collate_fn: Callable  # 데이터로더에서 배치를 만드는 함수
):
    """
    훈련을 시작하는 함수입니다. K-Fold 교차 검증을 사용하여 모델을 훈련합니다.

    Parameters:
    - kf (KFold object): K-fold 교차 검증을 위한 KFold 객체. 
      예: `kf = KFold(n_splits=5, shuffle=True, random_state=42)`
    - train_data (list/array): 훈련 데이터. 각 샘플의 특성(feature)을 포함하는 리스트.
    - target (list/array): 훈련 데이터에 대한 타겟 값 (라벨). 0 또는 1의 이진 값.
    - dataset (Dataset class): PyTorch Dataset 클래스를 상속한 데이터셋 클래스.
      예: `train_dataset = MyDataset(train_data, target)`
    - batch_size (int): 배치 크기. 한 번의 훈련에서 사용되는 데이터 샘플 수.
    - Net (class): 학습할 모델의 클래스. 
      예: `model = Net(**hp).to(device)`
    - hp (dict): 하이퍼파라미터 딕셔너리. 모델의 구성에 필요한 하이퍼파라미터들.
      예: `{"d_model": 512, "nhead": 8, "num_encoder_layers": 6, "num_decoder_layers": 6}`
    - device (str): 훈련에 사용할 장치. `"cpu"` 또는 `"cuda"` (GPU).
    - epochs (int): 훈련할 에폭(반복) 수.
    - loss_fn (torch.nn.Module): 손실 함수. 예: `torch.nn.CrossEntropyLoss()`
    - collate_fn (Callable): 데이터로더에서 배치를 생성하는 함수.

    Returns:
    - None
    """
    from model import Net
    from data import Dataset
    from torch.nn.utils.rnn import pad_sequence

    train_data = [torch.tensor(seq) for seq in train_data]
    train_data = pad_sequence(train_data, batch_first=True, padding_value=0)
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train)):
        print(f"Training fold {fold + 1}/{kf.get_n_splits()}")
        #train_data = np.array(train_data)
        #print(train_data[:2],type(train_data))
        # 학습용 데이터로더 객체
        train_dt = Dataset(train_data[train_idx], target[train_idx])
        train_dl = torch.utils.data.DataLoader(train_dt, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        # 검증용 데이터로더 객체
        valid_dt = Dataset(train_data[valid_idx], target[valid_idx])
        valid_dl = torch.utils.data.DataLoader(valid_dt, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        # 모델 객체와 옵티마이저 객체 생성
        model = Net(**hp).to(device)
        optimizer = torch.optim.Adam(model.parameters())

        best_score = 0  # 현재 최고 점수
        patience = 0  # 조기 종료 조건을 주기 위한 변수
        patience_threshold = 10000  # 최대 patience 값 (하이퍼파라미터로 설정)

        for epoch in tqdm(range(epochs)):
            # 훈련 단계
            model.train()
            train_loss = train_loop(train_dl, model, loss_fn, optimizer, device)

            # 검증 단계
            model.eval()
            valid_loss, pred = test_loop(valid_dl, model, loss_fn, device)

            # 이진 분류 문제에서 클래스 번호 결정 (0 또는 1)
            pred = (pred > 0.5).astype(int)

            # 정확도 계산
            score = accuracy_score(target[valid_idx], pred)

            # 성능 향상 시, 최고 점수 및 patience 초기화
            if score > best_score:
                best_score = score
                patience = 0
            else:
                patience += 1

            # 조기 종료 기준 (patience)
            if patience >= patience_threshold:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            #print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}, Valid Loss: {valid_loss}, Accuracy: {score}")
        
        # Save the best model
        #torch.save(model.state_dict(), f"best_model_fold{fold+1}.pt")
        
        print(f"{fold}th's Best Score: {best_score:.4f}")

    print("Training complete!")
