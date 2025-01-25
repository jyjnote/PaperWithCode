import pandas as pd
import numpy as np
import torch
import torch.utils
from tqdm.auto import tqdm
import random
import os
import cv2

def reset_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

SEED = 42
device = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_PATH = r'C:\\PapersWithCode\\03_Deep_Learning\\'
batch_size = 32

def main():
    reset_seeds(SEED)
    print(f'Using device: {device}')
    
    # DataFrame을 사용하여 열에 접근합니다.
    train = pd.read_csv(DATA_PATH + "/data/meat/train/class_info.csv")
    train_path = (DATA_PATH+"/data/meat/train/" + train["filename"]).to_numpy()

    #print(train_path)
    target = train["target"].to_numpy()

    from data_processing import MeatDataset,train_transform

    # MeatDataset을 생성
    train_dataset = MeatDataset(train_path, target,train_transform)
    # DataLoader 생성
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #print(next(iter(train_dl)))

    from train_test import start_training
    from model import Net,model_name

    model=Net(model_name,num_classes=3)

    start_training(model,train,
                   train_path,target,
                   batch_size,device
                   )

if __name__ == '__main__':
    main()
