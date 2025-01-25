from torchvision.models import resnet50, ResNet50_Weights
import torch


class FreezNet2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_model = resnet50(weights= ResNet50_Weights.IMAGENET1K_V2)
        
        for name, param in self.pre_model.named_parameters():
            if name.startswith("layer3"):
                break
            param.requires_grad = False # 가중치 업데이트 안함.

        self.pre_model.fc = torch.nn.Linear(2048, 1)

    def forward(self, x):
        return self.pre_model(x)