import torch
import numpy as np
def train_loop(model,loss_fn,dataloader,device,optimize):
    model.train()
    epoch_loss=0

    for batch in dataloader:
        preds=model(batch['x'].to(device))
        loss=loss_fn(preds, batch['y'].to(device))

        optimize.zero_grad()
        loss.backward()
        optimize.step()

        epoch_loss+=loss.item()

    return epoch_loss/len(dataloader)

@torch.no_grad()
def test_loop(model,dataloader,loss_fn,device):
    model.eval()
    epoch_loss=0
    preds_list=[]
    act_func=torch.nn.Sigmoid()

    for batch in dataloader:
        preds=model(batch['x'].to(device))

        if batch.get('y') is not None:
            loss=loss_fn(preds, batch['y'].to(device))
            epoch_loss+=loss.item()

        preds=act_func(preds).to('cpu').numpy()
        preds_list.append(preds)
    
    epoch_loss/=len(dataloader)
    preds=np.concatenate(preds_list)
    
    return epoch_loss, preds


def start_training(train,target,ReviewDataset):
    from sklearn.model_selection import KFold,StratifiedKFold
    from sklearn.metrics import accuracy_score
    from model import Net

    kf=KFold(n_splits=5, shuffle=True, random_state=42)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    n_features = train.shape[2]
    epochs=100
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i, (train_index, valid_index) in enumerate(kf.split(train)):
        print(f'Fold {i+1}/{kf.n_splits}')

        train_x,train_y=train[train_index],target[train_index]
        valid_x,valid_y=train[valid_index],target[valid_index]

        model=Net(n_features,15,1).to(device)
        optimizer=torch.optim.Adam(model.parameters())

        train_dataset=ReviewDataset(train_x, train_y)
        valid_dataset=ReviewDataset(valid_x, valid_y)

        train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=32, shuffle=True)
        valid_loader=torch.utils.data.DataLoader(valid_dataset,batch_size=32, shuffle=False)
        
        best_score=-np.inf
        patience=0
        
        for epoch in range(epochs):
            train_loss=train_loop(model, loss_fn, train_loader, device, optimizer)
            valid_loss, pred=test_loop(model, valid_loader, loss_fn, device)

            pred=(pred > 0.5).astype(int)
            score=accuracy_score(valid_y, pred)

            if score > best_score:
                best_score = score
            else:
                patience+=1

            if patience > 10:
                break

            #print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Accuracy: {score:.4f}')
        
        print(f'Best Score: {best_score:.4f}')

