
def main():
    from data import data_load
    data_path=r"C:\PapersWithCode\04_NLP\data\gpt\chat_dataset.csv"
    df=data_load(data_path)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_name = "skt/kogpt2-base-v2"
    model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                              bos_token='</s>',  # start 토큰
                              eos_token='</s>',  # end 토큰
                              unk_token='<unk>',
                              pad_token='<pad>',
                              mask_token='<mask>',
                              max_len = 1024)
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    loss_fn = torch.nn.CrossEntropyLoss()
    epochs = 10

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

    from data import train_loop,ChatDataset,CollateFN
    from model import chatbot
    
    train_dt = ChatDataset(df)
    collate_fn = CollateFN(tokenizer)
    train_dl = torch.utils.data.DataLoader(train_dt, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    

    for i in range(epochs):
        loss = train_loop(train_dl, model, loss_fn, optimizer, device)
        print(i, "번째: ", loss)

    max_len=100
    chatbot(model, tokenizer, max_len, device)

if __name__ == '__main__':
    main()