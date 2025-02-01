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
# DataLoader 설정 / 배치사이즈 32
train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
test_loader = DataLoader(testset, batch_size=32, shuffle=False)

# 데이터셋의 첫 번째 이미지와 라벨 확인
data_iter = iter(train_loader)  # DataLoader 객체를 반복 가능한 객체로 변환
images, labels = next(data_iter)  # 첫 번째 배치를 가져오기

