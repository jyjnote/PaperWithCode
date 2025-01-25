import torch

class Conv2Net(torch.nn.Module):
    def __init__(self, in_channel, out_channel,kernel_size):
        super().__init__()
        self.seq=torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, out_channel, kernel_size),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.seq(x)
    

class Net(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=16, kernel_size=3):
        super().__init__()
        self.seq = torch.nn.Sequential(
            Conv2Net(in_channels, out_channels, kernel_size),
            Conv2Net(out_channels, out_channels*2, kernel_size),
            Conv2Net(out_channels*2, out_channels*4, kernel_size),
            Conv2Net(out_channels*4, out_channels*8, kernel_size),
            torch.nn.AdaptiveMaxPool2d(1), # 글로벌 풀링, shape : batch, channel, 1, 1
            torch.nn.Flatten(), # 배치를 제외하고 평평하게, shape: batch, channel
            torch.nn.Dropout(0.2),
            torch.nn.Linear(out_channels*8, 1) # output layer
        )

    def forward(self, x):
        return self.seq(x)
    