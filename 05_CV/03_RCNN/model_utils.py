# model_utils.py
import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
from sklearn.linear_model import SGDClassifier, SGDRegressor
from torch.utils.data import DataLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_feature_extractor(device='cuda'):
    """
    VGG16 모델의 마지막 FC 레이어 이전까지를 feature extractor로 사용.
    """
    vgg16 = models.vgg16(pretrained=True)
    # 마지막 FC 레이어 제거 -> 4096차원 feature
    vgg16.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-1])
    vgg16.eval()
    vgg16.to(device)
    return vgg16

def train_rcnn_sgd(dataset, batch_size=64, n_epochs=5):
    """
    PyTorch DataLoader로 배치 단위 crop -> GPU -> VGG16 feature 추출 후,
    scikit-learn의 SGDClassifier와 4개의 SGDRegressor(좌표별)로 partial_fit.
    """
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = get_feature_extractor(device)

    # SGDClassifier: 확률적 경사 하강법(Stochastic Gradient Descent) 기반 선형 분류기
    # - loss='hinge': 힌지 손실을 사용하여 SVM 방식의 마진 최적화를 수행
    # - penalty='l2': L2 정규화를 적용하여 과적합을 방지 (가중치 제곱합에 패널티 부여)
    # - max_iter=1: 데이터셋에 대해 한 번만 순회(한 에포크)하며 학습; 이후 여러 번 partial_fit 등을 통해 점진적 업데이트 가능
    # - warm_start=True: 이전 학습 결과(가중치)를 유지하여 추가 학습 시 초기값으로 사용, 모델을 계속 업데이트할 수 있도록 함
    cls = SGDClassifier(loss='hinge', penalty='l2', max_iter=1, warm_start=True)
    # 4차원 타겟 처리를 위해 각 좌표마다 개별 regressor 생성
    reg_models = [SGDRegressor(penalty='l2', max_iter=1, warm_start=True) for _ in range(4)]

    all_labels = [dataset[i][1] for i in range(len(dataset))]
    classes = np.unique(all_labels)

    for epoch in range(n_epochs):
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
        for batch_img, batch_lbl, batch_box in pbar:
            batch_img = batch_img.to(device)
            with torch.no_grad():
                feat = feature_extractor(batch_img)  # (B, 4096)
            feat = feat.cpu().numpy()
            batch_lbl = np.array(batch_lbl)
            batch_box = np.array(batch_box)  # shape: (B, 4)

            cls.partial_fit(feat, batch_lbl, classes=classes)
            # 각 좌표에 대해 개별 partial_fit
            for j in range(4):
                reg_models[j].partial_fit(feat, batch_box[:, j])
    return cls, reg_models, feature_extractor

def evaluate_and_visualize(test_images, test_entries, test_data_list,
                           cls_model, reg_models, feature_extractor,
                           transform=None, num_images=5):
    """
    test_images, test_entries: build_rcnn_entries()로 얻은 테스트 이미지 및 proposal entry
    test_data_list: 원본 테스트 데이터셋(ground truth 정보 포함)
    cls_model, reg_models, feature_extractor: 학습된 모델들
    transform: RCNNProposalsDataset와 동일한 이미지 전처리 transform (미설정 시 기본값 사용)
    num_images: 시각화할 이미지 수
    """
    from collections import defaultdict
    if transform is None:
        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # 이미지별 proposal grouping (entry 내 img_id 기준)
    proposals_by_image = defaultdict(list)
    for entry in test_entries:
        proposals_by_image[entry['img_id']].append(entry)

    device = next(feature_extractor.parameters()).device
    selected_ids = list(proposals_by_image.keys())[:num_images]

    for img_id in selected_ids:
        # 원본 이미지 (BGR)
        img = test_images[img_id].copy()
        proposals = proposals_by_image[img_id]
        if len(proposals) == 0:
            continue

        feats = []
        valid_entries = []
        # 각 proposal에 대해 feature 추출
        for entry in proposals:
            x, y, w, h = entry['region']
            crop = img[y:y+h, x:x+w, :]
            if crop.size == 0:
                continue
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            from PIL import Image
            pil_img = Image.fromarray(crop_rgb)
            tensor_img = transform(pil_img).unsqueeze(0)  # (1, 3, 224, 224)
            tensor_img = tensor_img.to(device)
            with torch.no_grad():
                feat = feature_extractor(tensor_img)  # (1, 4096)
            feats.append(feat.cpu().numpy().squeeze(0))
            valid_entries.append(entry)

        if len(feats) == 0:
            continue
        feats = np.array(feats)
        # 분류 모델 예측 (hinge loss이므로 decision_function 사용 가능)
        scores = cls_model.decision_function(feats)
        preds = cls_model.predict(feats)
        # pedestrian (클래스 1)로 예측한 proposal 선택
        valid_idx = [i for i, p in enumerate(preds) if p == 1]
        if len(valid_idx) == 0:
            continue
        valid_scores = scores[valid_idx]
        valid_feats = feats[valid_idx]
        valid_entries = [valid_entries[i] for i in valid_idx]

        # 각 valid proposal에 대해 회귀 예측 (좌표별)
        pred_boxes = []
        for feat in valid_feats:
            coords = []
            for j in range(4):
                coords.append(reg_models[j].predict(feat.reshape(1, -1))[0])
            pred_boxes.append(coords)

        # 높은 score 순 상위 5개의 proposal 선택
        sorted_idx = np.argsort(valid_scores)[-5:]
        top_pred_boxes = [pred_boxes[i] for i in sorted_idx]

        # 원본 이미지에 GT 박스(녹색)와 예측 박스(빨간색) 그리기
        gt_item = test_data_list[img_id]
        gt_boxes = gt_item['boxes']
        for box in gt_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for box in top_pred_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Matplotlib을 이용해 시각화 (BGR -> RGB 변환)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 6))
        plt.imshow(img_rgb)
        plt.title(f"Test Image ID: {img_id}")
        plt.axis('off')
        plt.show()
