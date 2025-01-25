def train_loop(dataloader,model,loss_fn,optimizer,device):
    from tqdm import tqdm
    import torch

    epoch_loss = 0
    model.train()

    for batch in tqdm(dataloader):
        src=batch["src"].to(device)
        trg=batch["trg"].to(device)

        pred,_,_=model(src,trg)

        n_class=pred.shape[-1]
        pred = pred.view(-1, n_class)
        trg = trg.flatten()

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