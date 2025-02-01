import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import os
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
def train(model, train_loader, test_loader, num_epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # CUDA 사용 가능하면 GPU 사용, 아니면 CPU 사용
    model = model.to(device)  # 모델을 지정된 장치(GPU/CPU)로 이동
    
    criterion = nn.CrossEntropyLoss()  # 다중 클래스 분류 손실 함수
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 모델 학습 및 검증
    for epoch in range(num_epochs):
        model.train()  # 모델을 학습 모드로 전환
        running_loss = 0.0
        
        # 미니 배치 단위로 데이터를 불러옴
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 입력 데이터를 모델에 통과시킴
            if model.aux_logits:  # aux_logits가 True일 때
                outputs = model(inputs)  # 보조 출력을 포함한 모델 통과
                if isinstance(outputs, tuple):  # 모델이 튜플을 반환하면 보조 출력도 포함
                    outputs, aux1 = outputs  # 보조 출력이 하나만 반환되는 경우
                    aux1 = aux1.view(-1, 10)
                    loss1 = criterion(aux1, targets)
                    loss = criterion(outputs, targets) + 0.3 * loss1
                else:  # 보조 출력을 하나만 반환할 경우
                    outputs = outputs[0]  # 주 출력만 사용
                    targets = targets.view(-1)  # (batch_size * height * width,)
                    outputs = outputs.view(-1, 10)
                    loss = criterion(outputs, targets)
            else:  # aux_logits가 False일 때
                outputs = model(inputs)
                targets = targets.view(-1)  # (batch_size * height * width,)
                outputs = outputs.view(-1, 10)  # (batch_size * height * width, num_classes)
                loss = criterion(outputs, targets)

            # 역전파 및 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        
        # 검증 데이터로 성능 평가
        evaluate(model, test_loader, device)

# 검증 함수
def evaluate(model, test_loader, device):
    model.eval()  # 모델을 평가 모드로 전환
    correct = 0
    total = 0
    # 그라디언트 계산 비활성화
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # 입력 데이터를 지정된 장치로 이동
            outputs = model(inputs)
            outputs = outputs.view(-1, 10)  # (batch_size * height * width, num_classes)
            targets = targets.view(-1)  # (batch_size * height * width,)
            
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")






# In[ ]:


# 모델 생성 및 학습
model = GoogLeNet(aux_logits=True, num_classes=10)  # CIFAR10에는 10개의 클래스를 사용합니다
train(model, train_loader, test_loader, num_epochs=10, learning_rate=0.001)

