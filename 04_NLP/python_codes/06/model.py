import torch

class Net(torch.nn.Module):
    def __init__(self,input_size,hidden_size, num_classes):
        super().__init__()
        self.rnn_layer = torch.nn.LSTM(input_size, hidden_size,bidirectional=True, batch_first=True)
        self.hidden_layer = torch.nn.Linear(hidden_size*2, hidden_size//2)
        self.act_layer = torch.nn.ReLU()
        self.output_layer = torch.nn.Linear(hidden_size // 2, num_classes)

    def forward(self,x):
        outputs, (hn, cn) = self.rnn_layer(x)
        
        #print(hn.shape)

        hn=hn.permute(1,0,2)
        cn=cn.permute(1,0,2)

        #print(hn.shape)

        hn=hn.flatten(1)
        cn=cn.flatten(1)
        
        #print(hn.shape)

        x = self.hidden_layer(hn*cn) # hn[-1] and cn[-1] 주의하기
        x = self.act_layer(x)
        x = self.output_layer(x)
        #print(x) # 각 배치 방향으로의 100분에 몇 %인지 출력함
        #torch.nn.functional.softmax(x, dim=1) 
        #CrossEntropyLoss는 내부적으로 softmax를 계산하므로, 모델의 출력층에서 softmax를 제거하고 원시 로짓을 반환하도록 수정
        return  x