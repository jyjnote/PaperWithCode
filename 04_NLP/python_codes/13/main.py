import torch

DATA_PATH=r'C:\PapersWithCode\04_NLP\data\en2ko\translate_en_ko.csv'
def main():
    from data import dataload,TranslateDataset,collate_fn
    src_ko_data,trg_eng_data=dataload(DATA_PATH)
    
    from model import Net
    device="cuda" if torch.cuda.is_available() else "cpu"

    model = Net(len(src_ko_data), len(trg_eng_data), device=device).to(device)
    optimizer = torch.optim.Adam( model.parameters() )

    dt = TranslateDataset(src_ko_data, trg_eng_data)
    dl = torch.utils.data.DataLoader(dt, batch_size=64, shuffle=True, collate_fn=collate_fn)

    from train import train_loop

    loss_fn=torch.nn,CrossEntropyLoss()

    for epoch in range(50):
        epoch_loss = train_loop(dl, model, loss_fn, optimizer, device)
        print(epoch, epoch_loss)

if __name__=='__main__':
    main()