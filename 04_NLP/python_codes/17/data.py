

def data_load(DATA_PATH):
    import pandas as pd
    df = pd.read_csv(DATA_PATH)
    return df


def data_processing(data):
    data["ko"] = data["ko"].str.replace("[^가-힣0-9 .,!?\"\']", "" , regex=True)
    data["en"] = data["en"].str.replace("[^a-zA-Z0-9 .,!?\"\']", "" , regex=True).str.lower()

    from kiwipiepy import Kiwi
    kiwi = Kiwi()

    gen=kiwi.tokenize(data["ko"])
    src_data=[]
    for tokens in gen:
        tokens=[t.form for t in tokens]
        src_data.append(tokens)

    from torchtext.vocab import build_vocab_from_iterator
    vocab_src = build_vocab_from_iterator(src_data, specials=["<pad>", "<unk>"])
    vocab_src.set_default_index( vocab_src["<unk>"] )

    src_data = [ vocab_src(lst) for lst in src_data ]

    from torchtext.data.utils import get_tokenizer
    tokenizer = get_tokenizer("basic_english")
    
    sos_token = "<sos>"
    eos_token = "<eos>"

    trg_data = []
    for text in data["en"]:
        lst = [sos_token] + tokenizer(text) + [eos_token]
        trg_data.append(lst)

    vocab_trg = build_vocab_from_iterator(trg_data, specials=["<pad>", "<unk>", sos_token, eos_token])
    vocab_trg.set_default_index( vocab_trg["<unk>"] )

    trg_data = [ vocab_trg(lst) for lst in trg_data ]

    return src_data, trg_data,vocab_src,vocab_trg,gen


import torch

class TranslateDataset(torch.utils.data.Dataset):
    def __init__(self, src, trg):
        self.src = src
        self.trg = trg
    def __len__(self):
        return len(self.src)
    def __getitem__(self, idx):
        return self.src[idx], self.trg[idx]
    

def collate_fn(lst):
    src_lst = []
    trg_lst = []
    for src, trg in lst:
        src_lst.append( torch.tensor(src) )
        trg_lst.append( torch.tensor(trg) )

    src = torch.nn.utils.rnn.pad_sequence(src_lst, batch_first=True)
    trg = torch.nn.utils.rnn.pad_sequence(trg_lst, batch_first=True)

    return {"src": src, "trg": trg}