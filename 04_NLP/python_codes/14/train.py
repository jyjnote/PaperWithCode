
import torch
def train_loop(dataloader, model, loss_fn, optimizer, device):
    epoch_loss = 0
    model.train()
    
    for batch in dataloader:
        src = batch["src"].to(device)
        trg = batch["trg"].to(device)
        pred, _, _, _ = model(src,trg)

        # pred: batch, seq, n_class -> batch x seq, n_class
        n_class = pred.shape[-1] # 정답 클래스의 개수
        pred = pred.view(-1, n_class)

        #print(pred.shape)

        # trg: batch, seq -> batch x seq
        trg = trg.flatten()

        # pad, unk, sos 토큰 제외하고 손실 계산하기 위해 마스킹
        mask = trg > 2
        trg = trg[mask]
        pred = pred[mask]
        loss = loss_fn(pred, trg)

        optimizer.zero_grad() # 경사값 0으로 초기화
        loss.backward() # 역전파
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1) # 기울기 폭주 현상을 개선
        optimizer.step() # 가중치 업데이트

        epoch_loss += loss.item()

    epoch_loss /= len(dataloader)

    return epoch_loss