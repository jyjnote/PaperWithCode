import torch

class Net(torch.nn.Module):
    def __init__(self, vocab_sizes, embedding_dim,hidden_dim,out_features):
        super(Net, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_sizes, embedding_dim)
        self.rnn = torch.nn.LSTM(embedding_dim, hidden_dim*2, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim*2, out_features)


    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden,cell_state) = self.rnn(embedded)
        hidden=hidden.permute(1,0,2).flatten(1)
        out = self.fc(hidden)
        return out

