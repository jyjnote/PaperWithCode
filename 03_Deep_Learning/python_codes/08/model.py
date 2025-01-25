import timm
import torch

model_name = "wide_resnet50_2"

class Net(torch.nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        # 사전 학습된 모델 로드
        self.pre_model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        

    def forward(self, x):
        return self.pre_model(x)
    
    # 보통은 마지막 레이어층만 동결을 하지 않는다.