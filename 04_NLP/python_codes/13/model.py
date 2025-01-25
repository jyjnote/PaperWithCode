import torch

class Encoder(torch.nn.Module):
    def __init__(self, src_vocab_size, hidden_size):
        super().__init__()
        self.emb_layer = torch.nn.Embedding(src_vocab_size, hidden_size)
        self.lstm_layer = torch.nn.LSTM(hidden_size, hidden_size * 2, batch_first=True, bidirectional=True)

    def forward(self, src):
        # src: Input sequence of shape (batch_size, seq_length)
        src = self.emb_layer(src)  # Shape: (batch_size, seq_length, hidden_size)
        outputs, (hidden, cell) = self.lstm_layer(src)
        # outputs: Shape (batch_size, seq_length, hidden_size * 2) due to bidirectional
        # hidden, cell: Shape (num_layers * 2, batch_size, hidden_size)
        return outputs, hidden, cell

    
class Decoder(torch.nn.Module):
    def __init__(self, trg_vocab_size, hidden_size):
        super().__init__()
        self.emb_layer = torch.nn.Embedding(trg_vocab_size, hidden_size)
        self.lstm_layer = torch.nn.LSTM(hidden_size, hidden_size*4, batch_first=True)         #! 디코더의 lstm 크기는 임베딩의 hn,cn의 크기와 같아야함
        self.fc_layer = torch.nn.Linear(hidden_size*4, trg_vocab_size) # 예측하는

    def forward(self,trg, hn, cn): # trg는 하나의 시점의 텐서
        trg = self.emb_layer(trg) # batch, 1 -> batch, 1, features

# nlayer, batch, features -> 
# batch, nlayer, features -> 
# batch, nlayer x features ->
# 1, batch, nlayer x features
        hn = hn.permute(1,0,2).flatten(1).unsqueeze(0)
        cn = cn.permute(1,0,2).flatten(1).unsqueeze(0)
        _, (hn, cn)= self.lstm_layer(trg, (hn, cn) )

        # hn: nlayer, batch, features
        pred = self.fc_layer(hn[-1]) # 인덱싱해서 다음과 같은 텐서가 입력으로 전달: batch, features
        return pred, hn, cn
    

class Net(torch.nn.Module):
    def __init__(self, vocab_src_size, vocab_trg_size, hidden_size=64, device="cpu"):
        super().__init__()
        self.encoder = Encoder(vocab_src_size, hidden_size)
        self.decoder = Decoder(vocab_trg_size, hidden_size)
        self.device = device
        self.vocab_trg_size = vocab_trg_size # 타겟의 단어사전갯수 == 정답 클래스 개수
    def forward(self,src, trg, hn=None, cn=None):
        import random

        # trg: batch, seq
        batch_size = trg.shape[0]
        trg_len = trg.shape[1] 

        # prediction: batch, seq, n_class
        prediction = torch.zeros(batch_size, trg_len, self.vocab_trg_size).to(self.device)

        if hn is None:
            _,hn, cn = self.encoder(src)

        # 디코더에 전달 되는 첫번째 토큰데이터 == sos 토큰
        dec_input = trg[:, 0].view(-1,1)  # batch -> batch,seq  [2,0]->[[2],[0]]
        
        # if len(trg_len[0][0])==1:
        #    print("디코더의 첫 인풋값이 잘못되었습니다.")

        for t in range(1, trg_len): # 맞춰야하는 문장 길이만큼 예측 시작
            
            # pred : batch, n_class
            pred, hn, cn = self.decoder(dec_input, hn, cn)

            prediction[:,t] = pred # t 시퀀스의 예측 단어를 t시점에 넣어줌

            dec_input = pred.argmax(1).view(-1,1) # batch -> batch, seq

            if random.random() < 0.5: # 교사강요
                dec_input = trg[:,t].view(-1,1) # batch -> batch,seq
        
        #-----> 추론 부분에선 하나하나 예측을 하는것이다.
    
        return prediction, hn, cn