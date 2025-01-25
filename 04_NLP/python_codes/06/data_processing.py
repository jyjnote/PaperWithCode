import pandas as pd
import re 
from tqdm import tqdm
import numpy as np
import torch

re_pat = re.compile(r"\b\w{2,}\b")

class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, x, y=None):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return {'x': torch.tensor(self.x[idx],dtype=torch.float32), 
                'y': torch.tensor(self.y[idx],dtype=torch.float32) if self.y is not None else None}
        
def tokenizer(text, re_pat, stop_words) -> list:
    words = []
    text = text.lower() # 소문자화
    for w in re_pat.findall(text): # 정규표현식에 매칭되는 단어 리스트에 담아서 반환
        if w not in stop_words: # 불용어 제거
            words.append(w) # 단어 토큰 append
    return words

def data_processing():
    
    data=r'C:\PapersWithCode\04_NLP\data\imdb_dataset.csv'
    data=pd.read_csv(data)
    target = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0 ).to_numpy().reshape(-1,1)

    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    docs = []

    for text in data["review"]:
        doc = tokenizer(text, re_pat, ENGLISH_STOP_WORDS)
        docs.append(doc)

    from gensim.models.word2vec import Word2Vec
    emb_model = Word2Vec(docs, vector_size=64, sg=1, seed=42, min_count=1)
    len_list = [ len(doc) for doc in docs]
    max_len = np.mean(len_list).astype(int)

    train=[]
    for doc in tqdm(docs):
        vec=[emb_model.wv[w] for w in doc if emb_model.wv.key_to_index.get(w) ]
        vec=np.array(vec)[:max_len]

        if vec.shape[0]<max_len:
            diff=max_len - vec.shape[0]

            vec=np.concatenate([vec,np.random.random((diff,64))])

        train.append(vec)
    train=np.array(train)

    return train,target