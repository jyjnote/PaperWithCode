import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
def main():
    from data_processing import load_data,make_en_dict,make_ko_dict, TranslateDataset, collate_fn
    from model import Net
    

    ko, en = load_data()
    src_data=make_ko_dict(ko)
    trg_data =make_en_dict(en)

    model = Net(len(src_data), len(trg_data), device = device).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    train_dt = TranslateDataset(src_data, trg_data)
    train_dl = torch.utils.data.DataLoader(train_dt, batch_size=64, shuffle=True, collate_fn=collate_fn)

    from train_test import train_loop

    loss_fn=torch.nn.CrossEntropyLoss()

    for _ in range(50):
        train_loss = train_loop(train_dl, model, loss_fn, optimizer, device)
        print(train_loss)


if __name__ == "__main__":
    main()