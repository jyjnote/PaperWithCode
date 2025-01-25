import torch

class Net(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size, num_classes):
        super().__init__()
        self.emb_layer = torch.nn.Embedding(vocab_size,hidden_size)

        self.conv1d_block = torch.nn.Sequential(
            torch.nn.Conv1d(hidden_size, hidden_size*2, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),

            torch.nn.Conv1d(hidden_size*2, hidden_size, kernel_size=3, stride=1, padding=1),
            torch.nn.ELU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.rnn_layer = torch.nn.GRU(hidden_size, hidden_size*8, bidirectional=True,batch_first=True)

        self.output_layer = torch.nn.Linear(hidden_size*8*2, 4) # output layer

    def forward(self, x):
        x = self.emb_layer(x)
        x = x.permute(0,2,1) # b, s, f -> b, f, s
        x = self.conv1d_block(x) # b , f, s
        x = x.permute(0,2,1) # b , f, s -> b , s, f
        _, hn = self.rnn_layer(x) # hn : nlayer, b, f
        hn=hn.permute(1,0,2).flatten(1)

        return self.output_layer(hn)