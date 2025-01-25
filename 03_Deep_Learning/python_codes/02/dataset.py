import torch
import torch.utils.data
from data_processing import load_and_process_data  # data_processing.py에서 함수 가져오기
import numpy as np

# 데이터 로드 및 전처리
train_ft, test_ft, target = load_and_process_data()

class TitanicDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

        # y가 None이 아닌 경우, reshape 및 텐서 변환
        if self.y is not None:
            self.y = self.y.reshape(-1, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.y is not None:
            return {
                "x": torch.tensor(self.x[idx], dtype=torch.float32),
                "y": torch.tensor(self.y[idx], dtype=torch.float32)
            }
        else:
            return {
                "x": torch.tensor(self.x[idx], dtype=torch.float32),
            }


















