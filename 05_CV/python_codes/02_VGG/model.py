# pytorch 버전: 2.5.1
import torch

class A(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = torch.nn.ModuleList([
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        self.conv2 = torch.nn.ModuleList([
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        self.conv3 = torch.nn.ModuleList([
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        self.conv4 = torch.nn.ModuleList([
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        self.conv5 = torch.nn.ModuleList([
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        self.fc = torch.nn.Sequential(
            # FC-4096
            torch.nn.Linear(512*7*7, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            # FC-4096
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            # FC-C
            torch.nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        for layer in self.conv1: x = layer(x)
        for layer in self.conv2: x = layer(x)
        for layer in self.conv3: x = layer(x)
        for layer in self.conv4: x = layer(x)
        for layer in self.conv5: x = layer(x)
        x = self.fc(torch.flatten(x,1))
        return x

class B(A):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        # conv1 수정
        self.conv1.insert(len(self.conv1)-1, torch.nn.Conv2d(64, 64, kernel_size=3, padding=1))
        self.conv1.insert(len(self.conv1)-1, torch.nn.ReLU(inplace=True))
        # conv2 수정
        self.conv2.insert(len(self.conv2)-1, torch.nn.Conv2d(128, 128, kernel_size=3, padding=1))
        self.conv2.insert(len(self.conv2)-1, torch.nn.ReLU(inplace=True))

class C(B):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        # conv3 수정
        self.conv3.insert(len(self.conv3)-1, torch.nn.Conv2d(256, 256, kernel_size=1))
        self.conv3.insert(len(self.conv3)-1, torch.nn.ReLU(inplace=True))
        # conv4 수정
        self.conv4.insert(len(self.conv4)-1, torch.nn.Conv2d(512, 512, kernel_size=1))
        self.conv4.insert(len(self.conv4)-1, torch.nn.ReLU(inplace=True))
        # conv5 수정
        self.conv5.insert(len(self.conv5)-1, torch.nn.Conv2d(512, 512, kernel_size=1))
        self.conv5.insert(len(self.conv5)-1, torch.nn.ReLU(inplace=True))

# VGG-16
class D(C):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        # conv3 수정
        self.conv3[4] = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # conv4 수정
        self.conv4[4] = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # conv5 수정
        self.conv5[4] = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)

# VGG-19
class E(D):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        # conv3 수정
        self.conv3.insert(len(self.conv3)-1, torch.nn.Conv2d(256, 256, kernel_size=3, padding=1))
        self.conv3.insert(len(self.conv3)-1, torch.nn.ReLU(inplace=True))
        # conv4 수정
        self.conv4.insert(len(self.conv4)-1, torch.nn.Conv2d(512, 512, kernel_size=3, padding=1))
        self.conv4.insert(len(self.conv4)-1, torch.nn.ReLU(inplace=True))
        # conv5 수정
        self.conv5.insert(len(self.conv5)-1, torch.nn.Conv2d(512, 512, kernel_size=3, padding=1))
        self.conv5.insert(len(self.conv5)-1, torch.nn.ReLU(inplace=True))