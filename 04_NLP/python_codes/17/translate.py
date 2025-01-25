import torch

@torch.no_grad()
def translate(text, model, vocab_src, vocab_trg, max_len, device, tokenizer):
    model.eval()
    src = vocab_src([ t.form for t in tokenizer.tokenize(text)]) # 토큰화 -> 단어 번호 부여
    src = torch.tensor(src).view(1, -1).to(device) # 텐서로 변환후 배치차원 추가, batch(1), seq

    trg = [2] # 스타트 토큰
    trg = torch.tensor(trg).view(1, -1).to(device) # 텐서로 변환후 배치차원 추가, batch(1), seq

    memory = model.encoder(src) # batch, seq, features

    for _ in range(max_len):
        pred = model.decoder(trg, memory) # batch(1), seq, n_class

        token_no = pred[-1, -1].argmax().item()
        if token_no == 3:
            break

        print(  vocab_trg.lookup_token(token_no) , end=" ")


        next_token = torch.tensor([token_no]).view(1,-1).to(device) # 배치차원 추가, batch, seq
        trg = torch.cat([trg, next_token], dim=1)