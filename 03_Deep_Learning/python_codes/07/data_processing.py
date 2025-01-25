from torchvision import transforms
from PIL import Image

lst = [
    transforms.Resize([224,224]), # 사전학습 모델의 input 이미지 사이즈에 맞춰서 리사이징
    transforms.ToTensor(), # C, H, W 순서와  0~1범위로 스케일링
    transforms.Normalize( [0.485, 0.456, 0.406] , [0.229, 0.224, 0.225] ) # 사전학습 모델이 사용한 평균과 표준편차로 정규화
]
train_transform = transforms.Compose(lst)


lst = [
    transforms.Resize([224,224]), # 사전학습 모델의 input 이미지 사이즈에 맞춰서 리사이징
    transforms.ToTensor(), # C, H, W 순서와  0~1범위로 스케일링
    transforms.Normalize( [0.485, 0.456, 0.406] , [0.229, 0.224, 0.225] ) # 사전학습 모델이 사용한 평균과 표준편차로 정규화
]
test_transform = transforms.Compose(lst)

import torch

#class CatDogDataset(torch.utils.data.Dataset):
#    def __init__(self,x,y=None,transform=None):
#        self.x = x
#        self.y=y
#        self.transform =train_transform
#
#    def __len__(self):
#        return len(self.x)
#    
#    def __getitem__(self, index):
#        return {'x':self.transform(Image.open(self.x[index])) ,
#                'y' : torch.tensor(self.y[index],dtype=torch.float32)
#                if self.y is not None else None} 
    


class CatDogDataset(torch.utils.data.Dataset):
    def __init__(self,transform, x, y = None):
        self.transform = transform
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)

    def __getitem__(self,idx):
        item = {}
        x = Image.open(self.x[idx]) # 필로우 이미지 객체 반환
        item["x"] = self.transform(x) # 텐서로 변환
        if self.y is not None:
            item["y"] = torch.Tensor(self.y[idx]) # 텐서로 변환
        return item