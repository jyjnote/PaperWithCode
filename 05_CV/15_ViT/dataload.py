import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data


import torchvision 
from torchvision.datasets import CIFAR10 
from torchvision import transforms
import pytorch_lightning as pl

DATASET_PATH = "../data" 

# 테스트용 변환: 이미지를 텐서로 변환하고 정규화
test_transform = transforms.Compose([
    transforms.ToTensor(),  # 이미지를 텐서로 변환
    transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])  # 정규화 (평균, 표준편차)
])

# 훈련용 변환: 데이터 증강을 추가하여 과적합을 방지
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 50% 확률로 이미지를 수평으로 뒤집기 (데이터 증강)
    transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),  # 이미지를 랜덤 크기로 잘라내고 크기 비율을 조정
    transforms.ToTensor(),  # 이미지를 텐서로 변환
    transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])  # 정규화 (평균, 표준편차)
])
def data_load():
    train_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=train_transform, download=True)  # 훈련 데이터셋
    val_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=test_transform, download=True)  # 검증 데이터셋
    
    pl.seed_everything(42)
    train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000])
    
    pl.seed_everything(42)
    _, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000])  

    # 테스트 데이터셋 로딩
    test_set = CIFAR10(root=DATASET_PATH, train=False, transform=test_transform, download=True)  # 테스트 데이터셋

    # 훈련, 검증, 테스트를 위한 데이터 로더 설정
    train_loader = data.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)  # 훈련 데이터 로더
    val_loader = data.DataLoader(val_set, batch_size=128, shuffle=False, drop_last=False, num_workers=4)  # 검증 데이터 로더
    test_loader = data.DataLoader(test_set, batch_size=128, shuffle=False, drop_last=False, num_workers=4)  # 테스트 데이터 로더


    return train_loader, val_loader, test_loader