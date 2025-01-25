import torch
from torch.utils.data import Subset
# Training loop
def train_loop(model, dataloader, loss_fn, device, optimizer):
    model.train()
    epoch_loss = 0
    for batch in dataloader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        preds = model(inputs)
        loss = loss_fn(preds, targets)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

# Test loop
@torch.no_grad()
def test_loop(model, dataloader, loss_fn, device):
    import numpy as np
    model.eval()
    epoch_loss = 0
    act_func = torch.nn.Softmax(dim=1)
    pred_list = []
    
    for batch in dataloader:
        inputs, targets = batch
        inputs = inputs.to(device)
        
        preds = model(inputs)
        loss = loss_fn(preds, targets.to(device))
        epoch_loss += loss.item()
        
        preds = act_func(preds)
        preds = preds.cpu().numpy()
        pred_list.append(preds)
    
    epoch_loss /= len(dataloader)
    preds = np.concatenate(pred_list)
    
    return epoch_loss, preds

# Training and validation function
def start_train(training_data, validation_data, model, epochs=10):
    from sklearn.model_selection import KFold
    import numpy as np
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    

    early_stop_counter = 0
    patience = 5  # Early stopping patience
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, _) in enumerate(kf.split(training_data)):
        print(f'Fold {fold+1}')
        
        # Get the training data for the current fold
        train_subset = torch.utils.data.Subset(training_data, train_idx)
        train_dataloader = DataLoader(train_subset, batch_size=16, shuffle=True)
        
        # Validation data is fixed, no need to split it
        val_dataloader = DataLoader(validation_data, batch_size=16, shuffle=False)
        
        best_loss = np.inf
        best_epoch = 0
        
        for epoch in tqdm(range(epochs)):
            train_loss = train_loop(model, train_dataloader, loss_fn, device, optimizer)
            val_loss, _ = test_loop(model, val_dataloader, loss_fn, device)
            
            print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
            
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch + 1
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        print(f'k {fold}th {best_loss}')
            