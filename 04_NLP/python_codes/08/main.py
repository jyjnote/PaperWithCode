import pandas as pd
from data_processing import tokensize
import torch


DATA_PATH=r'C:\PapersWithCode\04_NLP\data\news\train_news.csv'

def main():
    data=pd.read_csv(DATA_PATH)
    train=data["desc"]
    target=data["target"].to_numpy()

    vocab_size,train_data=tokensize(train)
    
    from train_test import train_test
    train_test(vocab_size,train_data,target)
    
if __name__ == '__main__':
    main()    