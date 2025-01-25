def load_data():
    import pandas as pd
    data=pd.read_csv(r'C:\PapersWithCode\04_NLP\data\en2ko\translate_en_ko.csv')
    data["ko"],data["en"]=data["ko"].str.replace("[^가-힣 0-9,.!?\"\']","", regex=True),data["en"].str.replace("[^a-zA-Z 0-9,.!?\"\']","", regex=True).str.lower()

    data["en"] = "\t" + data["en"] + "\n"

    return data["ko"],data["en"]


def tokenizer(text, char2id):  # 문자열과 단어번호 부여하는 딕셔너리 전달 받기
    return [ char2id[c] for c in text]

def make_ko_dict(ko):
    tmp = " ".join(ko.tolist()) # 모든 샘플 하나의 문자열로 연결
    tmp = sorted(set(tmp))
    id2char_ko = dict(enumerate(tmp,2)) # 0번은 <pad>, 1번은 <unk> 을 위해 단어번호는 2부터 시작
    id2char_ko[0] = "<pad>"
    id2char_ko[1] = "<unk>"
    char2id_ko = { v:k for k,v in id2char_ko.items() }

    return [ tokenizer(text, char2id_ko) for text in ko ]


def make_en_dict(en):
    tmp = " ".join(en.tolist()) # 모든 샘플 하나의 문자열로 연결
    tmp = sorted(set(tmp)) # 중복 제거후 정렬
    id2char_en = dict(enumerate(tmp,2)) # 0번은 <pad>, 1번은 <unk> 을 위해 단어번호는 2부터 시작
    id2char_en[0] = "<pad>"
    id2char_en[1] = "<unk>"
    char2id_en = { v:k for k,v in id2char_en.items() }

    return [ tokenizer(text, char2id_en) for text in en ]

import torch

class TranslateDataset(torch.utils.data.Dataset):
    def __init__(self, src, trg):
        self.src = src
        self.trg = trg

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        item = {}
        item["src"] = torch.tensor(self.src[idx]) # 임베딩으로 넣을 것이기에 다음 tensor를 사용함.
        item["trg"] = torch.tensor(self.trg[idx]) # 임베딩으로 넣을 것이기에 다음 tesnor를 사용함.
        return item

# dataset = [
#     {"src": torch.tensor([1, 2, 3, 4]), "trg": torch.tensor([5, 6, 7])},
#     {"src": torch.tensor([8, 9, 10]), "trg": torch.tensor([11, 12, 13, 14])},
#     {"src": torch.tensor([15, 16, 17, 18, 19]), "trg": torch.tensor([20, 21])}
# ]

# lst = [
#     {"src": torch.tensor([1, 2, 3, 4]), "trg": torch.tensor([5, 6, 7])},
#     {"src": torch.tensor([8, 9, 10]), "trg": torch.tensor([11, 12, 13, 14])},
#     {"src": torch.tensor([15, 16, 17, 18, 19]), "trg": torch.tensor([20, 21])}
# ]
def collate_fn(dataset):

    # print(lst)
    
    src = [ item["src"] for item in dataset]
    src = torch.nn.utils.rnn.pad_sequence(src, batch_first=True)

    trg = [ item["trg"] for item in dataset]
    trg = torch.nn.utils.rnn.pad_sequence(trg, batch_first=True)

    return {"src": src, "trg": trg}

# 데이터로더에 들어오는 소스데이터와 타켓데이터를 다음과 같이 패딩을 자동화 해줌