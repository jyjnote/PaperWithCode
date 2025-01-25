


def main():
    from data import read_data,processing_data
    import torch

    text=read_data()

    id2char,char2id,train=processing_data(text)

    vocab_size = len(id2char) # 단어 사전 크기

    embedding_dim = 64
    batch_size = 64
    loss_fn = torch.nn.CrossEntropyLoss()
    epochs = 100

    from data import GenDataset,collate_fn
    from model import Net
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Net(vocab_size,embedding_dim).to(device)
    optimizer = torch.optim.Adam( model.parameters() )

    
    train_dt = GenDataset(train)
    train_dl = torch.utils.data.DataLoader(train_dt, batch_size = batch_size, shuffle=True, collate_fn = collate_fn)
    
    from train import train_loop
    from tqdm import tqdm

    for _ in tqdm(range(epochs)):
        loss = train_loop(train_dl, model, loss_fn, optimizer, device)
        print(f'epoch: {_,+3}, loss: {loss:.4f}')

    from generator import text_generator,sampling_text_generator
    text_generator(train[10], model, id2char,400,device)

    print("="*100)
    print("sampling")
    temp_list = [None, 0.1, 0.5, 1.2, 2.2]
    
    for temp in temp_list:
        print(f"온도: {temp}")
        sampling_text_generator(train[10], model, id2char,400,device)
        print("="*100)
        print("\n\n")

if __name__ == '__main__':
    main()