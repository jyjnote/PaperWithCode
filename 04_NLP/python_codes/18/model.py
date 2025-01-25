import torch

class Net(torch.nn.Module):
    def __init__(self,
                 vocab_size, # 단어사전크기
                 max_len,  # 최대시퀀스길이
                 d_model=512, # d_model,헤드수
                 nhead=8, # 헤드수
                 dim_feedforward=2048, # 피드포워드 신경망 부분의 노드수
                 num_layers=60, # 인코더 레이어수
                 device="cpu"):
        super().__init__()

        self.emb_layer = torch.nn.Embedding(vocab_size, d_model) # 단어 임베딩

        # 포지셔널 임베딩
        self.pos = torch.arange(max_len).to(device)
        self.pos_emb_layer = torch.nn.Embedding(max_len, d_model) # 위치정보 임베딩

        # 인코더 레이어
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        # 인코더
        self.encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers) # b, s, f

        # self.flatten = torch.nn.Flatten() # b, s x f
        # self.dropout = torch.nn.Dropout(0.5)
        self.gl_pool = torch.nn.AdaptiveMaxPool1d(1)
        self.flatten = torch.nn.Flatten()

        self.fc_out = torch.nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.emb_layer(x) # 단어 임베딩 b, s , f
        pos = self.pos_emb_layer(self.pos) # 위치 정보 임베딩 s, f
        x = x + pos # 배치 방향으로 브로드 캐스팅 되서 더하기 연산 된다. b, s, f
        x = self.encoder(x) # b, s ,f
        x = x.permute(0,2,1) # b , f , s
        x = self.gl_pool(x) # b, f, 1
        x = self.flatten(x) # b, f
        # x = self.dropout(x)
        return self.fc_out(x)
    


