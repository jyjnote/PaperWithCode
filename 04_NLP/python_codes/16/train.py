def train_loop(dataloader,model,loss_fn,optimizer,device):
    epoch_loss = 0
    model.train()

    for batch in dataloader:
        x,y = batch["x"].to(device),batch["y"].to(device)

        pred=model(x)

        loss=loss_fn(pred,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    epoch_loss /= len(dataloader)
    return epoch_loss