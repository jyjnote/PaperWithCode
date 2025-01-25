import torch
def train_loop(dataloader, model, loss_fn, optimizer, device):
    epoch_loss = 0
    model.train()

    for batch in dataloader:
        src = batch["src"].to(device)
        trg = batch["trg"].to(device) # trg: batch, seq (2,41)


        pred, _, _ = model(src, trg) # pred: batch, seq, 클래스별 예측값 (2,41,47)

        #print(f"pred: {pred.shape}") # pred: torch.Size([1, seq, 47]) seq는 다 다름

        num_class = pred.shape[-1] #(47)

        #print(f"num_class: {num_class}")

        pred = pred.view(-1, num_class) # batch x seq, 클래스별 예측값 (batch * seq_len, 47)

        #print(f"pred.view(-1, num_class): {pred.shape}")

        trg = trg.flatten() # batch x seq  (82)

        mask = trg > 2 # pad, unk, sos 토큰 제외하고 손실 계산하기 위한 마스킹
        trg = trg[mask]
        pred = pred[mask]

        #print(f"pred.view(-1, num_class) & trg: {pred.shape} {trg.shape}")
        
        # pred.view(-1, num_class) & trg: torch.Size([29, 47]) torch.Size([29])
        # pred.view(-1, num_class) & trg: torch.Size([seq, 47]) torch.Size([seq])
        loss = loss_fn(pred, trg) # 즉 타켓의 크기또한 배치*시퀀스의 크기만큼 존재하고 소프맥스함수의 의해 각 요소들이 오차값이 측정

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1) # 기울기 폭주 개선
        optimizer.step()

        epoch_loss += loss.item()
        #break

    epoch_loss /= len(dataloader)

    #print("===end_line==="*10)

    return epoch_loss