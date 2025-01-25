from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import torch

DATA_PATH=r"C:\PapersWithCode\04_NLP\data\review"
SEED=42
n_splits = 5
cv = KFold(n_splits, shuffle=True, random_state=SEED)

batch_size = 32 # 배치 사이즈
loss_fn = torch.nn.BCEWithLogitsLoss() # 손실 객체
epochs = 1000 # 최대 가능한 에폭수


def main():
    import warnings
    warnings.filterwarnings("ignore")   
    from data import data_loda,data_processing,Dataset,collate_fn
    train, target, test=data_loda(DATA_PATH)
    
    import torch
    train_data,test_list,vocab=data_processing(train, test)
    import numpy as np
    train=np.array(train['review'])
    max_len = max( len(lst) for lst in train_data )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #from data import build_vocab
    #vocab = build_vocab(train_data)

    hp = {
    "vocab_size":len(vocab),
    "max_len":max_len,
    "d_model":256,
    "nhead":8,
    "dim_feedforward":512,
    "num_layers":1,
    "device":device
    }
    # 0.5
    
    from sklearn.model_selection import KFold

    kf=KFold(n_splits=5,shuffle=True,random_state=42)
    #dataset=Dataset()

    from train import start_training
    batch_size=32

    #from model import Net
    #net=Net()

    #print(train_data[:5])
    #return
    start_training(kf,train_data,train,target,batch_size,
                   hp,device,epochs,loss_fn,collate_fn)
if __name__ == "__main__":
    main()