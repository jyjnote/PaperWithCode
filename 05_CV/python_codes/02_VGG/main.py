import torch
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.manual_seed(42)
else:
    device = torch.device('cpu')
    torch.manual_seed(42)

import data
from model import D
from train import train

if __name__ == '__main__':

    # ImageNet10 데이터 로드
    # train_dataset = data.load_train(path='ImageNet10/train')
    # valid_dataset = data.load_valid(path='ImageNet10/valid')
    from data import train_dataset,valid_dataset
    #! ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # 하이퍼파라미터 정의
    num_epochs    = 10
    batch_size    = 128
    learning_rate = 0.01
    momentum      = 0.9
    weight_decay  = 5e-4

    # 모델: VGG-16
    model = D(num_classes=10)
    # 옵티마이저: Momentum
    optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum, weight_decay)
    # 손실함수: Cross Entropy Loss
    criterion = torch.nn.CrossEntropyLoss()

    # 데이터 미니배치 처리
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size)

    # 훈련 시작
    train(model, train_loader, valid_loader, num_epochs, optimizer, criterion, device)