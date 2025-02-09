import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os

from dataset import get_dataloaders
from model import ResNet18

# 환경 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 모델 불러오기
net = ResNet18().to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True

# 데이터 로드
train_loader, test_loader = get_dataloaders()

# 하이퍼파라미터 설정
learning_rate = 0.1
num_epochs = 20
file_name = 'resnet18_cifar10.pt'

# 손실 함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002)

# 학습 함수
def train(epoch):
    print(f'\n[ Train Epoch: {epoch} ]')
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}: Loss {loss.item():.3f}, Accuracy {predicted.eq(targets).sum().item() / targets.size(0):.3f}')

    print(f'Total Train Accuracy: {100. * correct / total:.2f}%, Total Train Loss: {train_loss:.3f}')

# 평가 함수
def test(epoch):
    print(f'\n[ Test Epoch: {epoch} ]')
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(f'Test Accuracy: {100. * correct / total:.2f}%, Test Loss: {test_loss / total:.3f}')

    # 모델 저장
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save({'net': net.state_dict()}, f'./checkpoint/{file_name}')
    print('Model Saved!')

# 학습률 조정 함수
def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# 학습 시작
for epoch in range(num_epochs):
    adjust_learning_rate(optimizer, epoch)
    train(epoch)
    test(epoch)
