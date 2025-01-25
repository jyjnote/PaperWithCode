def make_ko_data(data):
    from kiwipiepy import Kiwi
    from tqdm import tqdm
    kiwi = Kiwi()

    map_instance = kiwi.tokenize(data)
    ko_src_data = []

    for tokens in tqdm(map_instance):
        tokens=[t.form for t in tokens]
        ko_src_data.append(tokens)
    
    from torchtext.vocab import build_vocab_from_iterator
    vocab_src=build_vocab_from_iterator(ko_src_data, specials=["<pad>","<unk>"])
    vocab_src.set_default_index(vocab_src["<unk>"])

    ko_src_data = [ vocab_src(tokens) for tokens in ko_src_data]

    return ko_src_data

def make_en_data(data):
    from torchtext.data.utils import get_tokenizer
    from torchtext.vocab import build_vocab_from_iterator

    tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

    sos_token = "<sos>"
    eos_token = "<eos>"

    en_trg_data = []
    for text in data:
        tokens = [sos_token] + tokenizer(text) + [eos_token]
        en_trg_data.append(tokens)

    vocab_trg = build_vocab_from_iterator(en_trg_data, specials=["<pad>", "<unk>", sos_token, eos_token])
    vocab_trg.set_default_index(vocab_trg["<unk>"])

    en_trg_data = [ vocab_trg(tokens) for tokens in en_trg_data ]

    return en_trg_data

import torch
class TranslateDataset(torch.utils.data.Dataset):
    def __init__(self, src, trg):
        self.src = src
        self.trg = trg

    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, idx):
        return {"src":torch.tensor(self.src[idx]),
                "trg":torch.tensor(self.trg[idx]) if self.trg is not None else None}

def collate_fn(dataset_list):
    from torch.nn.utils.rnn import pad_sequence

    src = [ item["src"] for item in dataset_list]
    src = pad_sequence(src, batch_first=True)

    trg = [ item["trg"] for item in dataset_list]
    trg = pad_sequence(trg, batch_first=True)

    return {"src":src,"trg":trg}