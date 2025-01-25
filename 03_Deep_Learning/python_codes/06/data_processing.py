SEED=42

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

lst = [
    A.Resize(224,224),
    A.HorizontalFlip(p=0.3),
    A.VerticalFlip(p=0.3),
    A.Affine(p=0.3),
    A.Normalize(),
    ToTensorV2()
]

train_transform = A.Compose(lst) # 학습용 변환 객체


lst = [
    A.Resize(224,224),
    A.Normalize(),
    ToTensorV2()
]

test_transform = A.Compose(lst) # 테스트 또는 검증용 변환 객체

import torch
import cv2
import os

class CatDogDataset(torch.utils.data.Dataset):
    def __init__(self, transform, x, y=None):
        self.transform = transform
        self.x = x
        self.target = y

    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        # 경로 확인 및 디버깅
        image_path = self.x[idx]

        if not isinstance(image_path, str):
            raise TypeError(f"Expected a string for image path, got {type(image_path)}")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image path does not exist: {image_path}")

        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found or cannot be opened: {image_path}")

        # BGR -> RGB 변환
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 변환 적용
        try:
            transformed = self.transform(image=image)['image']
        except Exception as e:
            raise RuntimeError(f"Error applying transform on image: {image_path}. Error: {e}") from e

        # 반환 값 생성
        return {
            'image': transformed,
            'target': self.target[idx] if self.target is not None else None
        }



# from glob import glob
# import os
# import numpy as np



# # 데이터 경로를 직접 설정합니다.
# DATA_PATH = r"C:\\PapersWithCode\\03_Deep_Learning\\data\\cats_and_dogs"  # 경로 수정

# # 고양이 이미지 파일 경로를 담은 리스트
# cats_list = sorted(glob(os.path.join(DATA_PATH, "train/cats/*.jpg")), key=lambda x: x)

# # 개 이미지 파일 경로를 담은 리스트
# dogs_list = sorted(glob(os.path.join(DATA_PATH, "train/dogs/*.jpg")), key=lambda x: x)

# # 각 리스트의 길이를 출력합니다.
# len(cats_list), len(dogs_list)


# labels = [0] * len(cats_list) + [1] * len(dogs_list) # 정답데이터 만들기
# img_path = cats_list + dogs_list # 고양이와 개 이미지 파일 경로 합치기


# # 멀티 인덱싱을 위해 ndarray 로 변환
# train = np.array(img_path)
# target = np.array(labels)

# np.random.seed(SEED) # 동일한 shuffle 위해 시드 고정

# # 인덱스를 이용하여 섞기 위해 샘플 개수 만큼 인덱스 생성
# index_arr = np.arange(train.shape[0])

# # 섞기
# np.random.shuffle(index_arr)
# np.random.shuffle(index_arr)

# # shuffle 된 인덱스를 이용하여 샘플 섞기
# train = train[index_arr]
# target = target[index_arr]

# test=CatDogDataset(train_transform,train,target)
# print(next(iter(test)))
