import torch
@torch.no_grad()
def text_generator(text, model, id2char, max_len,device):
    model.eval()
    x = torch.tensor(text).view(1,-1).to(device) # 배치차원 추가하기

    for _ in range(max_len):
        pred = model(x) # pred: batch(1), n_class
        char_no = pred.argmax(dim=1).item()
        
        print( id2char[char_no], end="" )

        # x : batch(1), seq
        add_tensor = torch.tensor([[char_no]]).to(device) # batch(1), seq(1)

        #print(add_tensor)

        x = torch.cat([ x[:,1:], add_tensor ], dim=1) 
        # 윈도우 단위로 한글텍스트를 다시 보내줘서 예측을 시작함

        #print(x)


@torch.no_grad()
def sampling_text_generator(text, model, id2char, max_len, device, temp=None):
    import numpy as np
    model.eval()
    x = torch.tensor(text).view(1,-1).to(device) # 배치차원 추가하기

    for _ in range(max_len):
        pred = model(x) # pred: batch(1), n_class
        char_no = pred.argmax(dim=1).item()

        if temp is not None:
            pred = pred[0] / temp  # 로짓값들을 소프트 맥스 온도로 나눠줌.
            max_ = pred.max()
            pred = pred - max_
            prob = torch.exp(pred) / torch.exp(pred).sum()                      
            # 소프트맥스 수식
            prob = prob.to("cpu").numpy()
            classes = np.arange(len(pred))
            char_no = np.random.choice(classes, 1, p=prob)[0]                   
            # 모든단어집(예측 클래스)에서의 하나를 뽑아야 하는데 이중 확률를 부여하여 뽑기의 다양성 추가

        print( id2char[char_no], end="" )

        # x : batch(1), seq
        add_tensor = torch.tensor([[char_no]]).to(device) # batch(1), seq(1)
        x = torch.cat([ x[:,1:], add_tensor ], dim=1)