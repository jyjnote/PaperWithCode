import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import pandas as pd

lst = [
    A.Resize(224,224),
    # A.HorizontalFlip(p=0.5),
    # A.VerticalFlip(p=0.5),
    A.Normalize(), # 스케일후 정규화 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ToTensorV2() # C ,H , W 순서의 텐서로 변환
]
train_transform = A.Compose(lst)

lst = [
    A.Resize(224,224),
    A.Normalize(), # 스케일후 정규화 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ToTensorV2() # C ,H , W 순서의 텐서로 변환
]
test_transform = A.Compose(lst)

import torch

class MeatDataset(torch.utils.data.Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x  # 이미지 경로 리스트
        self.y = y  # 레이블
        self.transform = transform  # 데이터 변환

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        # 이미지 로드 (경로에서 이미지를 읽음)
        image = cv2.imread(self.x[index])  # OpenCV로 이미지 읽기

        # 변환 적용 (albumentations는 'image=image' 형식으로 전달해야 함)
        if self.transform:
            image = self.transform(image=image)["image"]

        # 레이블은 None이 아닌 경우 텐서로 변환
        label = torch.tensor(self.y[index]) if self.y is not None else None

        return {'x': image, 'y': label}
    

DATA_PATH = r'C:\\PapersWithCode\\03_Deep_Learning\\'
train = pd.read_csv(DATA_PATH + "/data/meat/train/class_info.csv")
train_path = (DATA_PATH+"/data/meat/train/" + train["filename"]).to_numpy()

#print(train_path)
target = train["target"].to_numpy()
# MeatDataset을 생성
train_dataset = MeatDataset(train_path, target,train_transform)
# DataLoader 생성
train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=True)
print(next(iter(train_dl)))