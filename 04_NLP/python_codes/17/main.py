
DATA_PATH=r"C:\PapersWithCode\04_NLP\data\en2ko\translate_en_ko.csv"
def main():
    from data import data_load,data_processing,collate_fn,TranslateDataset
    df_data=data_load(DATA_PATH)
    src_data, trg_data,vocab_src,vocab_trg,gen=data_processing(df_data)

    hp = {
        "src_vocab_size": len(src_data),
        "trg_vocab_size": len(trg_data),
        "max_len" : 1000, # 시퀀스 최대길이
        "d_model" : 512,
        "nhead" : 8,
        "num_encoder_layers" : 1,
        "num_decoder_layers" : 1,
        "dim_feedforward" : 2048
      }

    import torch

    batch_size=64
    loss_fn = torch.nn.CrossEntropyLoss()
    epochs = 1000

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from model import Net
    epochs_list=[]
    model = Net(**hp).to(device)
    optimizer = torch.optim.Adam( model.parameters() )

    train_dt = TranslateDataset(src_data, trg_data)
    train_dl = torch.utils.data.DataLoader(train_dt, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    from tqdm import tqdm
    from train import train_loop
    
    for i in tqdm(range(epochs)):
        loss = train_loop(train_dl, model, loss_fn, optimizer, device)
        epochs_list.append(loss)
    
    print(f"{i}th training total_loss: {loss:.4f}")

    from translate import translate
    from kiwipiepy import Kiwi
    kiwi=Kiwi()

    translate('우리집 고양이는 천장을 보며 울어요.', model, vocab_src, vocab_trg, 100, device,kiwi)
    
if __name__ == '__main__':
    main()