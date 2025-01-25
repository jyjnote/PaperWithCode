# pytorch 버전: 2.5.1
import torch
from tqdm import tqdm
#! ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
def train(model, train_loader, valid_loader, num_epochs, optimizer, criterion, device):
    model.to(device)

    for epoch in range(num_epochs):
        print('Epoch {:0>3}    ==>'.format(epoch+1), end='    ')

        train_loss_list = []
        train_accu_list = []
        valid_loss_list = []
        valid_accu_list = []

        # 모델을 훈련 모드로 전환
        model.train()
        for image, label in tqdm(train_loader):
            #! ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            image, label = image.to(device), label.to(device)
            # 누적된 그래디언트를 0으로 초기화
            optimizer.zero_grad()
            # 순전파: 전방 계산
            output = model(image)
            # 순전파: 정확도 저장
            label_pred = torch.argmax(output, dim=1)
            train_accu_list += (label==label_pred).tolist()
            # 순전파: Loss 저장
            loss = criterion(output, label)
            train_loss_list.append(loss.item())
            # 역전파: 그래디언트 계산
            loss.backward()
            # 역전파: 가중치 갱신
            optimizer.step()

        # 모델을 평가 모드로 전환
        model.eval()
        with torch.no_grad():
            for image, label in valid_loader:
                image, label = image.to(device), label.to(device)
                # 전방 계산
                output = model(image)
                # 정확도 저장
                label_pred = torch.argmax(output, dim=1)
                valid_accu_list += (label==label_pred).tolist()
                # Loss 저장
                loss = criterion(output, label)
                valid_loss_list.append(loss.item())

        # 성능지표 계산
        train_loss = sum(train_loss_list) / len(train_loss_list)
        train_accu = sum(train_accu_list) / len(train_accu_list)
        valid_loss = sum(valid_loss_list) / len(valid_loss_list)
        valid_accu = sum(valid_accu_list) / len(valid_accu_list)
        print('train_loss: {:.4f}'.format(train_loss), end='  ')
        print('train_accu: {:.4f}'.format(train_accu), end='  ')
        print('valid_loss: {:.4f}'.format(valid_loss), end='  ')
        print('valid_accu: {:.4f}'.format(valid_accu))