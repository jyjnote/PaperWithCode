# PyTorch 버전: 2.5.1
import torch

# 기본 VGG 네트워크 클래스 (VGG-11 기반)
class A(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # 첫 번째 컨볼루션 블록 (Conv -> ReLU -> MaxPool)
        self.conv1 = torch.nn.ModuleList([
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 입력 채널 3 (RGB), 출력 채널 64
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 MaxPooling
        ])
        
        # 두 번째 컨볼루션 블록
        self.conv2 = torch.nn.ModuleList([
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 입력 64, 출력 128
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        
        # 세 번째 컨볼루션 블록
        self.conv3 = torch.nn.ModuleList([
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 입력 128, 출력 256
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 두 번째 Conv 레이어
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        
        # 네 번째 컨볼루션 블록
        self.conv4 = torch.nn.ModuleList([
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),  # 입력 256, 출력 512
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        
        # 다섯 번째 컨볼루션 블록
        self.conv5 = torch.nn.ModuleList([
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 입력 512, 출력 512
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        
        # 완전 연결층 (FC 레이어)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(512*7*7, 4096),  # 512채널, 7x7 크기의 입력을 받아 4096차원 출력
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),  # 드롭아웃 적용
            
            torch.nn.Linear(4096, 4096),  # 두 번째 FC 레이어
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            
            torch.nn.Linear(4096, num_classes)  # 최종 클래스 개수에 맞춰 출력
        )
    
    def forward(self, x):
        """순전파 함수: 정의된 컨볼루션 레이어를 순차적으로 통과"""
        for layer in self.conv1: x = layer(x)
        for layer in self.conv2: x = layer(x)
        for layer in self.conv3: x = layer(x)
        for layer in self.conv4: x = layer(x)
        for layer in self.conv5: x = layer(x)

        # Fully Connected 층에 전달하기 위해 flatten
        x = self.fc(torch.flatten(x, 1))
        return x

# VGG-13 구조 (Conv 레이어 추가)
class B(A):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        
        # conv1 블록에 64채널 Conv 추가
        self.conv1.insert(len(self.conv1)-1, torch.nn.Conv2d(64, 64, kernel_size=3, padding=1))
        self.conv1.insert(len(self.conv1)-1, torch.nn.ReLU(inplace=True))
        
        # conv2 블록에 128채널 Conv 추가
        self.conv2.insert(len(self.conv2)-1, torch.nn.Conv2d(128, 128, kernel_size=3, padding=1))
        self.conv2.insert(len(self.conv2)-1, torch.nn.ReLU(inplace=True))

# VGG-16 구조 (1x1 컨볼루션 추가)
class C(B):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        
        # conv3 블록에 1x1 컨볼루션 추가
        self.conv3.insert(len(self.conv3)-1, torch.nn.Conv2d(256, 256, kernel_size=1))
        self.conv3.insert(len(self.conv3)-1, torch.nn.ReLU(inplace=True))
        
        # conv4 블록에 1x1 컨볼루션 추가
        self.conv4.insert(len(self.conv4)-1, torch.nn.Conv2d(512, 512, kernel_size=1))
        self.conv4.insert(len(self.conv4)-1, torch.nn.ReLU(inplace=True))
        
        # conv5 블록에 1x1 컨볼루션 추가
        self.conv5.insert(len(self.conv5)-1, torch.nn.Conv2d(512, 512, kernel_size=1))
        self.conv5.insert(len(self.conv5)-1, torch.nn.ReLU(inplace=True))

# VGG-16 최종 수정 (기존 1x1 Conv를 3x3 Conv로 변경)
class D(C):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        
        # conv3 블록: 1x1 Conv를 3x3 Conv로 변경
        self.conv3[4] = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        # conv4 블록: 1x1 Conv를 3x3 Conv로 변경
        self.conv4[4] = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        
        # conv5 블록: 1x1 Conv를 3x3 Conv로 변경
        self.conv5[4] = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)

# VGG-19 (VGG-16에 Conv 추가)
class E(D):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        
        # conv3 블록에 추가적인 256채널 Conv 추가
        self.conv3.insert(len(self.conv3)-1, torch.nn.Conv2d(256, 256, kernel_size=3, padding=1))
        self.conv3.insert(len(self.conv3)-1, torch.nn.ReLU(inplace=True))
        
        # conv4 블록에 추가적인 512채널 Conv 추가
        self.conv4.insert(len(self.conv4)-1, torch.nn.Conv2d(512, 512, kernel_size=3, padding=1))
        self.conv4.insert(len(self.conv4)-1, torch.nn.ReLU(inplace=True))
        
        # conv5 블록에 추가적인 512채널 Conv 추가
        self.conv5.insert(len(self.conv5)-1, torch.nn.Conv2d(512, 512, kernel_size=3, padding=1))
        self.conv5.insert(len(self.conv5)-1, torch.nn.ReLU(inplace=True))
