from transformers import AutoTokenizer, AutoModel
import torch
def load_model(model_name):
    model = AutoModel.from_pretrained(model_name)
    return model

class Net(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.pre_model = AutoModel.from_pretrained(model_name)
        self.fc_out = torch.nn.Linear( self.pre_model.config.hidden_size, 1)

    def forward(self, x):
        x = self.pre_model(**x)
        # x[0]: 모든 시점의 히든출력 batch, seq, features
        # x[1]: CLS 토큰의 히든출력 batch, features
        return self.fc_out(x[1])