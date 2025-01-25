import torch
class Net(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.emb_layer = torch.nn.Embedding(vocab_size, embedding_dim)
        self.rnn_layer = torch.nn.LSTM(embedding_dim, embedding_dim*2, batch_first=True, bidirectional=True,num_layers=4)
        self.fc_out = torch.nn.Linear(embedding_dim*4*4, vocab_size)

    def forward(self, x): # x: batch ,seq
        x = self.emb_layer(x) # batch, seq, features
        _, (hn, _) = self.rnn_layer(x)
        # hn : nlayer, batch, features
        # nlayer, batch, features -> batch, nlayer, features -> batch, nlayer x features
        x = hn.permute(1,0,2).flatten(1)
        return self.fc_out(x)