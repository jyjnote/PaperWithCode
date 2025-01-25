
import numpy as np
import torch

def transform_data(data, seq_len = 10, pred_len = 5):
    mins = data.min(axis=0)
    sizes = data.max(axis=0) - mins
    data = (data - mins) / sizes # 스케일링
    x_list = []
    y_list = []
    for i in range(seq_len, data.shape[0]+1 - pred_len ):
        x = data[i-seq_len:i] # 입력 데이터
        y = data[i:i+pred_len,3] # 정답데이터

        x_list.append(x)
        y_list.append(y)

    x_arr = np.array(x_list)
    y_arr = np.array(y_list)

    return x_arr, y_arr,mins,sizes

class FinanceDataset(torch.utils.data.Dataset):
    def __init__(self, x_list, y_list=None):
        self.x_list = x_list
        self.y_list = y_list

    def __len__(self):
        return len(self.x_list)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.x_list[idx], dtype=torch.float32)
        if self.y_list is not None:
            y = torch.tensor(self.y_list[idx], dtype=torch.float32)

            return {'x':x,'y':y}
        
        else:
            return {'x':x}
        

# test=FinanceDataset(x_list,y_list)
# test=torch.utils.data.DataLoader(test)
# print(next(iter(test)))