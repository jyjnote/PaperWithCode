import torch
import torch.nn as nn
from data_processing import FinanceDataset,transform_data
import pandas_datareader.data as web


class Resnet_Seqblock_LstmNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Resnet_Seqblock_LstmNet, self).__init__()
        
        # Bidirectional LSTM layer
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        

        self.sq_block = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size * 2,hidden_size * 2)  
        )
        
        self.sq_block_list = nn.ModuleList([self.sq_block for _ in range(5)])
        
        # Linear layers to transform and output final predictions
        self.linear1 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.linear2 = nn.Linear(hidden_size * 2, 5)

    def forward(self, x):
        outputs, (hn, cn) = self.rnn(x)
        
        outputs_mean = outputs.mean(dim=1)
        
        #print(f"outputs_mean: {outputs_mean.shape}")
        
        hn=hn.permute(1,0,2).flatten(1)

        #print(f"hn_last: {hn.shape}")

        outputs_mean=self.linear1(outputs_mean)

        #print(f"outputs_mean: {outputs_mean.shape}")

        #print(f"hn * outputs_mean: {(hn * outputs_mean).shape}")
        for sq_block in self.sq_block_list:
            x = hn * outputs_mean  
            x = sq_block(x)
            #print(f"x: {x.shape}")
            
        x = self.linear2(x)
        

        #print(f"x: {x.shape}")

        return x


    
    

# df_1 = web.DataReader('005930', 'naver', start='2022-01-01', end='2022-12-31')
# df_1 = df_1.astype(int)
# data_1 = df_1.to_numpy()

# train_x_arr,train_y_arr=transform_data(data_1)

# test=FinanceDataset(train_x_arr,train_y_arr)
# test=next(iter(torch.utils.data.DataLoader(test,batch_size=10)))

# test_model=Resnet_Seqblock_LstmNet(train_x_arr.shape[-1],10)

# print(test_model(test['x']))
