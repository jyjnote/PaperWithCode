import torch


class Encoder(torch.nn.Module):
    def __init__(self, src_vocab_size, embedding_dim): # 어휘집 크기, 임베딩벡터 크기
        super().__init__()
        self.emb_layer = torch.nn.Embedding(src_vocab_size, embedding_dim)
        self.rnn_layer = torch.nn.LSTM(embedding_dim, embedding_dim*2, batch_first=True)

    def forward(self, x): # src 텐서가 전달
        x = self.emb_layer(x) # batch, seq, features

        # outputs: batch, seq, features
        # hn: nlayer, batch, features
        # cn: nlayer, batch, features
        outputs, (hn, cn) = self.rnn_layer(x)

        return hn, cn
    

class Decoder(torch.nn.Module):
    def __init__(self, trg_vocab_size, embedding_dim): # 어휘집 크기, 임베딩 벡터크기
        super().__init__()
        self.emb_layer = torch.nn.Embedding(trg_vocab_size, embedding_dim)
        # 디코더에 출력 히든 크기는 인코더에서 전달받은 히든과 셀스테이트의 피처 크기와 같아야 한다.
        
        self.rnn_layer = torch.nn.LSTM(embedding_dim, embedding_dim*2 , batch_first=True)

        self.fc_layer = torch.nn.Linear(embedding_dim*2, trg_vocab_size) # 단어 예측하는 layer

    def forward(self, x, hn, cn): # x 텐서: 하나의 시점에 해당하는 단어 텐서가 입력으로 전달됨.
        x = self.emb_layer(x) # b, 1 , f , 다음 처음 들어갈 단어는 시작을 알리는 스타트 토큰이 1번으로 들어감.


        # outputs: batch, 1, feature
        # hn : nlayer , batch, feature
        # cn : nlayer , batch, feature
        outputs, (hn, cn) = self.rnn_layer(x,(hn, cn)) # 여기에서 맨처음 hn,cn은 인코더에서 받은 정보임,

        pred = self.fc_layer(hn[-1]) # hn[-1]에 의미는: nlayer , batch, feature -> batch, feature

        return pred, hn, cn
    
class Net(torch.nn.Module):
    def __init__(self, vocab_size_src, vocab_size_trg, embedding_dim = 64, device = "cpu"):
        super().__init__()
        self.encoder = Encoder(vocab_size_src, embedding_dim) 
        # vocab_size_src는 한국어 단어의 총 갯수
        self.decoder = Decoder(vocab_size_trg, embedding_dim)
        # vocab_size_trg는 한국어 단어의 총 갯수

        self.vocab_size_trg = vocab_size_trg
        self.device = device
        
    def forward(self, src, trg, hn=None, cn=None): # trg shape: batch, seq
        import random
        # src, trg 은 실제 학습 혹 추론을 진행할 샘플 데이터

        batch_size = trg.shape[0]
        trg_len = trg.shape[1] # 예측해야하는 문장의 길이
        prediction = torch.zeros(batch_size, trg_len, self.vocab_size_trg).to(self.device) 
        # batch, seq 문장의 길이, 각 문장속의 단어들의 임베딩 값들///클래스 예측값 들

        # print(prediction)

        if hn is None: # 실제 데이터 예측시 모델을 반복 하기 때문에 hn 들어올수도 있어서 조건문으로 체크
            hn, cn = self.encoder(src)

        dec_input = trg[:,0].view(-1,1) # 각 샘플의 sos 토큰

        for t in range(1, trg_len): # 문장의 길이만큼 다음 단어들을 예측 시작

            # pred: batch, 클래스 예측값들
            # hn 과 cn : nlayer , batch, features
            pred, hn, cn = self.decoder(dec_input, hn, cn) # 현재시점 입력에 대한 예측

            prediction[:,t] = pred  # pred는 예측 단어임, 손실계산을 위해 예측데이터 저장
                                    # 디코더가 예측한 타켓 단어들의 각 맞을 확률값 모음
            # 디코더에 들어가는 단어번호 교체
            dec_input = pred.argmax(1).view(-1,1) 
            # batch -> batch, seq
            # dec_input = trg[:,0].view(-1,1)와 같이 변형 시켜줘야함

            # 교사강요
            if random.random() < 0.5:
                dec_input = trg[:,t].view(-1,1) # batch -> batch, seq

        return prediction, hn, cn