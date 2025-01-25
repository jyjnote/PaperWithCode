import torch
class PositionalEncoding(torch.nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        pos_encoding = torch.zeros(max_len, d_model) # seq, features

        #print(pos_encoding.shape)

        pos = torch.arange(0, max_len, dtype=torch.float32).view(-1,1) # seq, 1
        _2i = torch.arange(0, d_model, step=2, dtype=torch.float32) # d_model 절반 크기의 벡터

        pos_encoding[:, 0::2] = torch.sin( pos / 10000 ** (_2i/d_model) )
        pos_encoding[:, 1::2] = torch.cos( pos / 10000 ** (_2i/d_model) )

        #print(pos.shape)

        pos_encoding = pos_encoding.unsqueeze(0) # seq, features -> batch(1), seq, features

        #print(pos_encoding.shape)

        # 첫번째 인수로 인스턴스 변수명, 두번째 인수로 저장하고자하는 데이터 전달
        # cpu 또는 gpu 메모리에서 모두 작동
        # 학습 가능한 파라미터가 있는 텐서를 등록할 경우 업데이트를 안한다.
        self.register_buffer("pos_encoding", pos_encoding) # 텐서가 버퍼에 등록 됨!

    def forward(self,x): # x는 임베딩 텐서, batch , seq, features
        return x + self.pos_encoding[ :, :x.shape[1] ]
    
class Net(torch.nn.Module):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 max_len = 1000, # 시퀀스 최대길이
                 d_model = 512,
                 nhead = 8,
                 num_encoder_layers = 6,
                 num_decoder_layers = 6,
                 dim_feedforward = 2048):
        super().__init__()

        # 임베딩
        self.src_emb = torch.nn.Embedding(src_vocab_size, d_model)
        self.trg_emb = torch.nn.Embedding(trg_vocab_size, d_model)

        # 위치 인코딩
        self.pe = PositionalEncoding(max_len, d_model)

        # 트랜스 포머
        self.transformer = torch.nn.Transformer(d_model=d_model,
                                                nhead=nhead,
                                                num_encoder_layers=num_encoder_layers,
                                                num_decoder_layers=num_decoder_layers,
                                                dim_feedforward=dim_feedforward,
                                                batch_first=True)

        # 단어 예측
        self.fc_out = torch.nn.Linear(d_model, trg_vocab_size)

    def forward(self, src, trg, tgt_mask):

        src = self.pe( self.src_emb(src) ) # 임베딩 -> 위치정보반영 b, s ,f
        trg = self.pe( self.trg_emb(trg) ) # 임베딩 -> 위치정보반영 b, s, f

        x = self.transformer(src, trg, tgt_mask=tgt_mask) # b, s, f

        return self.fc_out(x) # batch, seq, n_class

    def encoder(self, src):
        src = self.pe( self.src_emb(src) ) # 임베딩 -> 위치정보반영 b, s ,f
        return self.transformer.encoder(src) # b, s ,f

    def decoder(self, trg, memory ):
        trg = self.pe( self.trg_emb(trg) ) # 임베딩 -> 위치정보반영 b, s, f
        x = self.transformer.decoder(trg, memory = memory) # b ,s ,f
        return self.fc_out(x) # batch, seq, n_class