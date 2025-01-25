from transformers import AutoTokenizer, AutoModel
import torch

class Net(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.fc_out = torch.nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, x):
        x = self.model(**x)
        return self.fc_out(x[1]) # cls 토큰 출력을 fc layer 전달하여 예측
