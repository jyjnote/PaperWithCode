import torch  # 추가
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    # 기본적인 Convolution + BatchNorm + ReLU 모듈 정의
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()

        # bias=False로 설정, BatchNorm이 bias 역할을 함
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs),  # Convolution layer
            nn.BatchNorm2d(out_channels),  # Batch Normalization
            nn.ReLU()  # ReLU activation
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Stem(nn.Module):
    # Inception-ResNet에서 첫 번째 Stem 블록을 정의 (입력 이미지의 크기를 변경)
    def __init__(self):
        super().__init__()

        # 첫 번째 Conv + Conv + Conv 블록 (이미지 크기를 점차적으로 변화)
        self.conv1 = nn.Sequential(
            BasicConv2d(3, 32, 3, stride=2, padding=0),  # 149x149x32
            BasicConv2d(32, 32, 3, stride=1, padding=0),  # 147x147x32
            BasicConv2d(32, 64, 3, stride=1, padding=1),  # 147x147x64
        )

        # 여러 가지 분기(branch)로 나누는 연산들
        self.branch3x3_conv = BasicConv2d(64, 96, 3, stride=2, padding=0)  # 73x73x96
        self.branch3x3_pool = nn.MaxPool2d(4, stride=2, padding=1)  # 73x73x64

        # 다양한 Conv + Conv 블록 (7x7 필터를 사용)
        self.branch7x7a = nn.Sequential(
            BasicConv2d(160, 64, 1, stride=1, padding=0),
            BasicConv2d(64, 96, 3, stride=1, padding=0)
        )  # 71x71x96

        self.branch7x7b = nn.Sequential(
            BasicConv2d(160, 64, 1, stride=1, padding=0),
            BasicConv2d(64, 64, (7, 1), stride=1, padding=(3, 0)),
            BasicConv2d(64, 64, (1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(64, 96, 3, stride=1, padding=0)
        )  # 71x71x96

        # 풀링 후에 채널을 확장
        self.branchpoola = BasicConv2d(192, 192, 3, stride=2, padding=0)  # 35x35x192
        self.branchpoolb = nn.MaxPool2d(4, 2, 1)  # 35x35x192


    def forward(self, x):
        x = self.conv1(x)  # Stem 부분 처리
        # 3x3 conv와 pooling 결과를 합침
        x = torch.cat((self.branch3x3_conv(x), self.branch3x3_pool(x)), dim=1)
        # 7x7a, 7x7b 결과를 합침
        x = torch.cat((self.branch7x7a(x), self.branch7x7b(x)), dim=1)
        # 최종적으로 두 가지 branch를 합침
        x = torch.cat((self.branchpoola(x), self.branchpoolb(x)), dim=1)
        return x


class Inception_Resnet_A(nn.Module):
    # Inception-ResNet-A 블록 정의 (Residual Connection 포함)
    def __init__(self, in_channels):
        super().__init__()

        self.branch1x1 = BasicConv2d(in_channels, 32, 1, stride=1, padding=0)

        self.branch3x3 = nn.Sequential(
            BasicConv2d(in_channels, 32, 1, stride=1, padding=0),
            BasicConv2d(32, 32, 3, stride=1, padding=1)
        )

        self.branch3x3stack = nn.Sequential(
            BasicConv2d(in_channels, 32, 1, stride=1, padding=0),
            BasicConv2d(32, 48, 3, stride=1, padding=1),
            BasicConv2d(48, 64, 3, stride=1, padding=1)
        )
        
        # 1x1 convolution을 통해 채널을 변환
        self.reduction1x1 = nn.Conv2d(128, 384, 1, stride=1, padding=0)
        self.shortcut = nn.Conv2d(in_channels, 384, 1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(384)  # Batch Normalization
        self.relu = nn.ReLU()  # ReLU 활성화 함수

    def forward(self, x):
        x_shortcut = self.shortcut(x)  # Short-cut 연결
        # 여러 개의 분기들을 합침
        x = torch.cat((self.branch1x1(x), self.branch3x3(x), self.branch3x3stack(x)), dim=1)
        # 1x1 convolution으로 채널 수를 줄임
        x = self.reduction1x1(x)
        # Residual Connection 추가 (shortcut + 연산 결과)
        x = self.bn(x_shortcut + x)
        x = self.relu(x)
        return x


class Inception_Resnet_B(nn.Module):
    # Inception-ResNet-B 블록 정의 (Residual Connection 포함)
    def __init__(self, in_channels):
        super().__init__()

        self.branch1x1 = BasicConv2d(in_channels, 192, 1, stride=1, padding=0)
        self.branch7x7 = nn.Sequential(
            BasicConv2d(in_channels, 128, 1, stride=1, padding=0),
            BasicConv2d(128, 160, (1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(160, 192, (7, 1), stride=1, padding=(3, 0))
        )

        self.reduction1x1 = nn.Conv2d(384, 1152, 1, stride=1, padding=0)
        self.shortcut = nn.Conv2d(in_channels, 1152, 1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(1152)  # Batch Normalization
        self.relu = nn.ReLU()  # ReLU 활성화 함수

    def forward(self, x):
        x_shortcut = self.shortcut(x)  # Short-cut 연결
        # 분기들 합침
        x = torch.cat((self.branch1x1(x), self.branch7x7(x)), dim=1)
        # 1x1 convolution으로 채널 수를 줄임
        x = self.reduction1x1(x) * 0.1
        # Residual Connection 추가
        x = self.bn(x + x_shortcut)
        x = self.relu(x)
        return x


class Inception_Resnet_C(nn.Module):
    # Inception-ResNet-C 블록 정의 (Residual Connection 포함)
    def __init__(self, in_channels):
        super().__init__()

        self.branch1x1 = BasicConv2d(in_channels, 192, 1, stride=1, padding=0)
        self.branch3x3 = nn.Sequential(
            BasicConv2d(in_channels, 192, 1, stride=1, padding=0),
            BasicConv2d(192, 224, (1, 3), stride=1, padding=(0, 1)),
            BasicConv2d(224, 256, (3, 1), stride=1, padding=(1, 0))
        )

        self.reduction1x1 = nn.Conv2d(448, 2144, 1, stride=1, padding=0)
        self.shortcut = nn.Conv2d(in_channels, 2144, 1, stride=1, padding=0)  # 2144
        self.bn = nn.BatchNorm2d(2144)  # Batch Normalization
        self.relu = nn.ReLU()  # ReLU 활성화 함수

    def forward(self, x):
        x_shortcut = self.shortcut(x)  # Short-cut 연결
        # 분기들 합침
        x = torch.cat((self.branch1x1(x), self.branch3x3(x)), dim=1)
        # 1x1 convolution으로 채널 수를 줄임
        x = self.reduction1x1(x) * 0.1
        # Residual Connection 추가
        x = self.bn(x_shortcut + x)
        x = self.relu(x)
        return x

    
class ReductionA(nn.Module):
    # Inception-ResNet에서 피처맵을 축소하는 ReductionA 블록
    def __init__(self, in_channels, k, l, m, n):
        super().__init__()

        self.branchpool = nn.MaxPool2d(3, 2)  # Pooling으로 피처맵 축소
        self.branch3x3 = BasicConv2d(in_channels, n, 3, stride=2, padding=0)  # 3x3 conv
        self.branch3x3stack = nn.Sequential(
            BasicConv2d(in_channels, k, 1, stride=1, padding=0),
            BasicConv2d(k, l, 3, stride=1, padding=1),
            BasicConv2d(l, m, 3, stride=2, padding=0)
        )

    def forward(self, x):
        # 세 가지 분기 결과를 합침
        x = torch.cat((self.branchpool(x), self.branch3x3(x), self.branch3x3stack(x)), dim=1)
        return x


class ReductionB(nn.Module):
    # Inception-ResNet에서 피처맵을 축소하는 ReductionB 블록
    def __init__(self, in_channels):
        super().__init__()

        self.branchpool = nn.MaxPool2d(3, 2)  # Pooling으로 피처맵 축소
        self.branch3x3a = nn.Sequential(
            BasicConv2d(in_channels, 256, 1, stride=1, padding=0),
            BasicConv2d(256, 384, 3, stride=2, padding=0)
        )
        self.branch3x3b = nn.Sequential(
            BasicConv2d(in_channels, 256, 1, stride=1, padding=0),
            BasicConv2d(256, 288, 3, stride=2, padding=0)
        )
        self.branch3x3stack = nn.Sequential(
            BasicConv2d(in_channels, 256, 1, stride=1, padding=0),
            BasicConv2d(256, 288, 3, stride=1, padding=1),
            BasicConv2d(288, 320, 3, stride=2, padding=0)
        )

    def forward(self, x):
        # 여러 가지 분기 결과를 합침
        x = torch.cat((self.branchpool(x), self.branch3x3a(x), self.branch3x3b(x), self.branch3x3stack(x)), dim=1)
        return x


class InceptionResNetV2(nn.Module):
    # 전체 Inception-ResNet-v2 모델 정의
    def __init__(self, A, B, C, k=256, l=256, m=384, n=384, num_classes=10, init_weights=True):
        super().__init__()
        blocks = []
        blocks.append(Stem())  # Stem 블록 추가
        for i in range(A):  # Inception-ResNet-A 블록 반복
            blocks.append(Inception_Resnet_A(384))
        blocks.append(ReductionA(384, k, l, m, n))  # ReductionA 추가
        for i in range(B):  # Inception-ResNet-B 블록 반복
            blocks.append(Inception_Resnet_B(1152))
        blocks.append(ReductionB(1152))  # ReductionB 추가
        for i in range(C):  # Inception-ResNet-C 블록 반복
            blocks.append(Inception_Resnet_C(2144))

        self.features = nn.Sequential(*blocks)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive Average Pooling
        self.dropout = nn.Dropout2d(0.2)  # 드롭아웃
        self.linear = nn.Linear(2144, num_classes)  # 최종 클래스 분류를 위한 선형 계층

        if init_weights:
            self._initialize_weights()  # 가중치 초기화

    def forward(self, x):
        x = self.features(x)  # 특징 추출
        x = self.avgpool(x)  # Adaptive Average Pooling
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)  # 드롭아웃
        x = self.linear(x)  # 선형 분류
        return x

    # 가중치 초기화 함수 정의
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # He Initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
