import torch

class NN(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.fc_layer1=torch.nn.Linear(n_features,8)
        self.batch_norm1=torch.nn.BatchNorm1d(8)
        self.relu1=torch.nn.ReLU()
        self.dropout1=torch.nn.Dropout(0.5)
        
        self.fc_layer2=torch.nn.Linear(8,4)
        self.batch_norm2=torch.nn.BatchNorm1d(4)
        self.relu2=torch.nn.ReLU()
        self.dropout2=torch.nn.Dropout(0.5)
        self.out_layer=torch.nn.Linear(4,1)

    def forward(self,x):
        x=self.fc_layer1(x)
        x=self.batch_norm1(x)
        x=self.relu1(x)
        x=self.dropout1(x)
        x=self.fc_layer2(x)
        x=self.batch_norm2(x)
        x=self.relu2(x)
        x=self.dropout2(x)
        x=self.out_layer(x)
        return x        
