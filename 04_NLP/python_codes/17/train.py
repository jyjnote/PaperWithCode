import torch

def train_loop(dataloader, model, loss_fn, optimizer, device):
    epoch_loss = 0
    model.train()
    
    for batch in dataloader:
        src = batch["src"].to(device) # batch, seq
        trg = batch["trg"].to(device) # batch, seq
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(trg.shape[1]).to(device)

        pred = model(src, trg, tgt_mask) # batch, seq, n_class

        n_class = pred.shape[-1] # 정답 클래스 개수
        # 예측값에서는 eos 토큰 입력에 대한 예측값을 제외
        pred = pred[:,:-1].reshape(-1, n_class) # batch x seq, n_class

        # 정답에서 sos 토큰 제외
        trg = trg[:,1:].flatten() # batch x seq

        mask = trg > 2
        trg = trg[mask]
        pred = pred[mask]
        loss = loss_fn(pred, trg)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(dataloader)

    return epoch_loss