
DATA_PATH = r"C:\PapersWithCode\04_NLP\data\en2ko\translate_en_ko.csv"

import torch
def main():
    import pandas as pd 
    from tqdm import tqdm

    data=pd.read_csv(f"{DATA_PATH}")
    ko,en=data["ko"].str.replace("[^가-힣 0-9,.!?\"\']","", regex=True),data["en"].str.replace("[^a-zA-Z 0-9,.!?\"\']","", regex=True).str.lower()

    from data import make_ko_data,TranslateDataset,collate_fn
    ko=make_ko_data(ko)
    en=make_ko_data(en)
    
    loss_fn = torch.nn.CrossEntropyLoss()

    from model import Net
    device="cuda" if torch.cuda.is_available() else "cpu"
    model = Net(len(ko), len(en), device=device).to(device)
    optimizer = torch.optim.Adam( model.parameters() )

    dt = TranslateDataset(ko, en)
    dl = torch.utils.data.DataLoader(dt, batch_size=64, shuffle=True, collate_fn=collate_fn)

    from train import train_loop
    for epoch in tqdm(range(50)):
        epoch_loss = train_loop(dl, model, loss_fn, optimizer, device)
        print(epoch, epoch_loss)

if __name__ == '__main__':
    main()