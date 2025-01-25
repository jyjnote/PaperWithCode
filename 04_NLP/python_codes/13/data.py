

def dataload(data_path):
    import pandas as pd

    train=pd.read_csv(data_path)

    train["ko"] = train["ko"].str.replace("[^가-힣 0-9,.!?\"\']","", regex=True)
    train["en"] = train["en"].str.replace("[^a-zA-Z 0-9,.!?\"\']","", regex=True).str.lower()

    from kiwipiepy import Kiwi

    kiwi = Kiwi()
    result = kiwi.tokenize(train["ko"])

    src_data = []
    
    from tqdm import tqdm

    for tokens in tqdm(result):
        tokens = [ t.form for t in tokens]
        src_data.append(tokens)

    from torchtext.vocab import build_vocab_from_iterator
    vocab_src=build_vocab_from_iterator(src_data,specials=["<pad>","<unk>"])
    vocab_src.set_default_index(vocab_src["<unk>"])

    src_ko_data = [ vocab_src(tokens) for tokens in src_data]
    
    # -----------------------------------eng-------------------------------------------

    from torchtext.data.utils import get_tokenizer
    tokenizer = get_tokenizer("basic_english")

    sos_token = "<sos>" # start of sentence
    eos_token = "<eos>" # end of sentence

    trg_data = []
    
    for text in train["en"]:
        tokens = [sos_token] + tokenizer(text) + [eos_token]
        trg_data.append(tokens)

    vocab_trg = build_vocab_from_iterator(trg_data, specials=["<pad>", "<unk>", sos_token, eos_token])
    vocab_trg.set_default_index(vocab_trg["<unk>"])
    
    trg_eng_data = [ vocab_trg(tokens) for tokens in trg_data ]


    return src_ko_data,trg_eng_data

import torch

class TranslateDataset(torch.utils.data.Dataset):
    def __init__(self, src, trg):
        self.src = src
        self.trg = trg
    def __len__(self):
        return len(self.src)
    def __getitem__(self, idx):
        item = {}
        item["src"] = torch.tensor( self.src[idx] )
        item["trg"] = torch.tensor( self.trg[idx] )
        return item
    
def collate_fn(dataset_lst):
    from torch.nn.utils.rnn import pad_sequence

    src = [ item["src"] for item in dataset_lst]
    src = pad_sequence(src, batch_first=True)

    trg = [ item["trg"] for item in dataset_lst]
    trg = pad_sequence(trg, batch_first=True)

    return {"src": src, "trg": trg}