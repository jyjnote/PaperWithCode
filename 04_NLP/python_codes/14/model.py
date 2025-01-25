
import torch

class Encoder(torch.nn.Module):
    def __init__(self, vocab_size_src, embedding_dim):
        super().__init__()
        self.emb_layer = torch.nn.Embedding(vocab_size_src, embedding_dim)
        self.lstm_layer = torch.nn.LSTM(embedding_dim, embedding_dim * 2, batch_first=True, bidirectional=True,dropout=0.4)
        self.embedding_dim = embedding_dim

    def forward(self, src):
        x = self.emb_layer(src) # batch, seq -> batch, seq, features
        out,(hidden,cell)=self.lstm_layer(x)
        hidden = hidden.permute(1,0,2).flatten(1).unsqueeze(0) #
        cell = cell.permute(1,0,2).flatten(1).unsqueeze(0) #

        return out,hidden,cell



class Decoder(torch.nn.Module):
    def __init__(self, vocab_size_trg, embedding_dim):
        super().__init__()
        self.emb_layer=torch.nn.Embedding(vocab_size_trg,embedding_dim)
        self.lstm_layer = torch.nn.LSTM(embedding_dim, embedding_dim * 4, batch_first=True,dropout=0.5)
        self.attn_key_layer = torch.nn.Linear(embedding_dim*4, embedding_dim*4)
        self.fc_layer = torch.nn.Linear(embedding_dim*8, vocab_size_trg)

    def forward(self, trg,enc_out, hidden, cell):
        trg=self.emb_layer(trg)
        _, (hidden, cell) = self.lstm_layer(trg, (hidden,cell) ) 
        attn_key = self.attn_key_layer(enc_out)
        attn_key = attn_key.permute(1,0,2).flatten(1)
        attn_query = hidden[-1].view(-1).unsqueeze(1)
        attn_scores = torch.matmul(attn_key, attn_query)
        attn_scores = torch.nn.functional.softmax(attn_scores, dim=0)
        attn_scores = attn_scores.view(1,1,-1).repeat(enc_out.shape[0], 1, 1)
        attn_values = torch.bmm(attn_scores, enc_out)
        x = torch.cat([ hidden[-1], attn_values[:,-1] ], dim=1)
        return self.fc_layer(x), hidden, cell
    
class Net(torch.nn.Module):
    def __init__(self, vocab_size_src, vocab_size_trg, embedding_dim=64, device="cpu"):
        super().__init__()
        self.encoder = Encoder(vocab_size_src, embedding_dim)
        self.decoder = Decoder(vocab_size_trg, embedding_dim)
        self.device = device
        self.vocab_size_trg = vocab_size_trg # 타겟의 단어사전갯수 == 정답 클래스 개수
    def forward(self,src, trg,enc_outputs=None, hidden=None, cell=None):
        import random

        # trg: batch, seq
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        # prediction: batch, seq, n_class
        prediction = torch.zeros(batch_size, trg_len, self.vocab_size_trg).to(self.device)

        if hidden is None:
            enc_outputs, hidden, cell = self.encoder(src)
        # 디코더에 전달 되는 첫번째 토큰데이터 == sos 토큰
        dec_input = trg[:, 0].view(-1,1)  # batch -> batch,seq

        for t in range(1, trg_len):

            # pred : batch, n_class
            pred, hidden, cell = self.decoder(dec_input, enc_outputs ,hidden, cell)

            prediction[:,t] = pred

            dec_input = pred.argmax(1).view(-1,1) # batch -> batch, seq

            if random.random() < 0.5: # 교사강요
                dec_input = trg[:,t].view(-1,1) # batch -> batch,seq

        return prediction,enc_outputs , hidden, cell