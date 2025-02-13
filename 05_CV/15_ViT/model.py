import os  # os 라이브러리를 임포트하여 디렉토리 작업을 수행
import urllib.request  # urllib 라이브러리에서 URL 요청을 위한 모듈을 임포트
from urllib.error import HTTPError  # HTTPError를 처리하기 위한 예외 모듈을 임포트
import torch

base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/"

# 다운로드할 파일 목록 정의 (비전 트랜스포머와 기타 체크포인트 파일)
pretrained_files = [
    "tutorial15/ViT.ckpt",  # Vision Transformer 체크포인트 파일
    "tutorial15/tensorboards/ViT/events.out.tfevents.ViT",  # 비전 트랜스포머의 TensorBoard 로그 파일
    "tutorial5/tensorboards/ResNet/events.out.tfevents.resnet"  # ResNet의 TensorBoard 로그 파일
]


def img_to_patch(x, patch_size, flatten_channels=True):
    """
    입력:
        x - 형태가 [B, C, H, W]인 이미지 텐서 (B: 배치 크기, C: 채널 수, H: 높이, W: 너비)
        patch_size - 패치의 각 차원에서 픽셀 수 (정수)
        flatten_channels - True일 경우, 패치들을 평평한 벡터로 반환 (이미지 그리드가 아닌 특징 벡터로 반환)
    
    출력:
        x - 패치로 분할된 이미지 텐서
    """
    B, C, H, W = x.shape  # 입력 텐서 x의 배치 크기(B), 채널 수(C), 높이(H), 너비(W) 추출
    x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)  # 이미지 텐서를 패치 크기에 맞게 재구성
    x = x.permute(0, 2, 4, 1, 3, 5)  # 텐서의 차원 순서를 변경하여 [B, H', W', C, p_H, p_W]로 만듦
    x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W] 형태로 평평하게 만듦
    if flatten_channels:
        x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W] 형태로 채널 차원을 평평하게 만듦
    return x

import torch.nn as nn
class AttentionBlock(nn.Module):
    
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network 
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()
        
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, 
                                          dropout=dropout)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        
    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x

class VisionTransformer(nn.Module):
    
    def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size, num_patches, dropout=0.0):
        """
        입력:
            embed_dim - 트랜스포머로 입력되는 특성 벡터의 차원 크기
            hidden_dim - 트랜스포머 내부의 피드포워드 네트워크에서 사용되는 숨겨진 레이어의 차원 크기
            num_channels - 입력 이미지의 채널 수 (RGB는 3)
            num_heads - Multi-Head Attention에서 사용할 헤드 수
            num_layers - 트랜스포머에서 사용할 레이어 수
            num_classes - 예측할 클래스 수
            patch_size - 패치의 크기 (각 차원의 픽셀 수)
            num_patches - 이미지에서 최대 패치 수
            dropout - 피드포워드 네트워크와 입력 인코딩에서 적용할 드롭아웃 비율
        """
        super().__init__()
        
        self.patch_size = patch_size
        
        # 레이어/네트워크
        self.input_layer = nn.Linear(num_channels*(patch_size**2), embed_dim)  # 입력 패치를 임베딩 차원으로 변환하는 선형 레이어
        self.transformer = nn.Sequential(*[AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])  # 트랜스포머 레이어들
        self.mlp_head = nn.Sequential(  # MLP 헤드 (최종 분류 예측을 위한 네트워크)
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)  # 드롭아웃 적용
        
        # 파라미터/임베딩
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))  # CLS 토큰 (분류를 위한 학습 가능한 파라미터)
        self.pos_embedding = nn.Parameter(torch.randn(1,1+num_patches,embed_dim))  # 위치 인코딩

    
    def forward(self, x):
        # 입력 전처리
        x = img_to_patch(x, self.patch_size)  # 이미지를 패치로 변환
        B, T, _ = x.shape  # B: 배치 크기, T: 패치 수
        x = self.input_layer(x)  # 패치를 임베딩 차원으로 변환
        
        # CLS 토큰과 위치 인코딩 추가
        cls_token = self.cls_token.repeat(B, 1, 1)  # CLS 토큰을 배치 크기만큼 복제
        x = torch.cat([cls_token, x], dim=1)  # CLS 토큰을 입력 시퀀스의 첫 번째에 추가
        x = x + self.pos_embedding[:,:T+1]  # 위치 인코딩 추가
        
        # 트랜스포머 적용
        x = self.dropout(x)  # 드롭아웃 적용
        x = x.transpose(0, 1)  # 트랜스포머에 입력할 때 차원 순서 변경 (배치 차원과 시퀀스 차원 전환)
        x = self.transformer(x)  # 트랜스포머 처리
        
        # 분류 예측 수행
        cls = x[0]  # CLS 토큰의 최종 출력값을 사용
        out = self.mlp_head(cls)  # MLP 헤드를 통해 분류 예측
        return out

























