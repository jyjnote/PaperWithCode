# main.py
import os
from sklearn.model_selection import train_test_split
from data_utils import load_penn_fudan_data, build_rcnn_entries, RCNNProposalsDataset
from model_utils import train_rcnn_sgd, evaluate_and_visualize

def main():
    root_dir = "PennFudanPed"  # 데이터셋 루트 경로
    data_list = load_penn_fudan_data(root_dir)
    print("전체 이미지 수:", len(data_list))
    
    # 학습/테스트 데이터 분할 (예: 80% 학습, 20% 테스트)
    train_data_list, test_data_list = train_test_split(data_list, test_size=0.2, random_state=42)
    print("학습 이미지:", len(train_data_list), "테스트 이미지:", len(test_data_list))
    
    # 각 데이터셋에 대해 R-CNN entry 생성
    train_images, train_entries = build_rcnn_entries(train_data_list)
    test_images, test_entries = build_rcnn_entries(test_data_list)
    print("학습 프로포절:", len(train_entries), "테스트 프로포절:", len(test_entries))
    
    # PyTorch Dataset 생성
    train_dataset = RCNNProposalsDataset(train_images, train_entries)
    test_dataset = RCNNProposalsDataset(test_images, test_entries)  # 평가에 직접 사용하지 않고 시각화용으로 활용
    
    # 학습 수행
    cls_model, reg_models, cnn_model = train_rcnn_sgd(train_dataset, batch_size=64, n_epochs=5)
    print("학습 완료.")
    
    # 테스트 결과 시각화 (예: 5개 이미지)
    evaluate_and_visualize(test_images, test_entries, test_data_list,
                           cls_model, reg_models, cnn_model, num_images=5)

if __name__ == "__main__":
    main()
