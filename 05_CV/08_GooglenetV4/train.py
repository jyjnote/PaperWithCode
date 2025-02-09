# define loss function and optimizer
# function to get current learning rate
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

# function to calculate metric per mini-batch
def metric_batch(output, target):
    pred = output.argmax(1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects

# function to calculate loss per mini-batch
def loss_batch(loss_func, output, target, opt=None):
    loss_b = loss_func(output, target)
    metric_b = metric_batch(output, target)

    if opt is not None:
        opt.zero_grad()
        loss_b.backward()
        opt.step()

    return loss_b.item(), metric_b

# function to calculate loss per epoch
def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None,device="cuda"):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)

    for xb, yb in dataset_dl:
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)

        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)

        running_loss += loss_b

        if metric_b is not None:
            running_metric += metric_b

        if sanity_check is True:
            break

    loss = running_loss / len_data
    metric = running_metric / len_data

    return loss, metric

# function to start training
def train_val(model, params):
    import torch
    import time
    from tqdm import tqdm
    from torch.utils.tensorboard import SummaryWriter  # TensorBoard 추가

    # 파라미터들 받아오기
    num_epochs = params['num_epochs']
    loss_func = params["loss_func"]
    opt = params["optimizer"]
    train_dl = params["train_dl"]
    val_dl = params["val_dl"]
    sanity_check = params["sanity_check"]
    lr_scheduler = params["lr_scheduler"]
    path2weights = params["path2weights"]

    loss_history = {'train': [], 'val': []}
    metric_history = {'train': [], 'val': []}

    best_loss = float('inf')

    # TensorBoard Writer 설정
    writer = SummaryWriter(log_dir=params.get('tensorboard_log', 'runs/experiment'))

    start_time = time.time()

    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs-1, current_lr))

        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, sanity_check, opt)
        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_metric)

        # TensorBoard에 훈련 손실과 정확도 기록
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', 100*train_metric, epoch)

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl, sanity_check)
        loss_history['val'].append(val_loss)
        metric_history['val'].append(val_metric)

        # TensorBoard에 검증 손실과 정확도 기록
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', 100*val_metric, epoch)

        if val_loss < best_loss:
            best_loss = val_loss
            print('Get best val_loss!')

        # Learning rate scheduler 업데이트
        lr_scheduler.step(val_loss)

        # 진행 상황 출력 (tqdm 사용)
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"train loss: {train_loss:.6f}, val loss: {val_loss:.6f}, "
              f"accuracy: {100*val_metric:.2f}%, time: {(time.time()-start_time)/60:.4f} min")
        print('-'*10)

    writer.close()  # TensorBoard Writer 종료

    return model, loss_history, metric_history
