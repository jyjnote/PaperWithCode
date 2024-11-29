# coding: utf-8
import numpy as np


def smooth_curve(x):
    """손실 함수의 그래프를 매끄럽게 하기 위해 사용
    
    참고：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]


def shuffle_dataset(x, t):
    """데이터셋을 뒤섞는다.

    Parameters
    ----------
    x : 훈련 데이터
    t : 정답 레이블
    
    Returns
    -------
    x, t : 뒤섞은 훈련 데이터와 정답 레이블
    """
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    t = t[permutation]

    return x, t

def conv_output_size(input_size, filter_size, stride=1, pad=0):
    return (input_size + 2*pad - filter_size) / stride + 1


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """다수의 이미지를 입력받아 2차원 배열로 변환한다(평탄화).
    
    Parameters
    ----------
    input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
        CNN 계층의 입력 데이터로, 다수의 이미지 데이터를 포함함.
    filter_h : 필터의 높이
        필터의 세로 크기.
    filter_w : 필터의 너비
        필터의 가로 크기.
    stride : int, optional
        필터를 적용할 때 이동하는 보폭의 크기. 기본값은 1.
    pad : int, optional
        입력 데이터에 추가할 패딩의 폭. 기본값은 0.
    
    Returns
    -------
    col : 2차원 배열
        입력 데이터가 필터와 합성곱 연산을 수행할 수 있도록 평탄화된 2차원 배열.
    """
    # 입력 데이터의 차원 (N: 이미지 수, C: 채널 수, H: 높이, W: 너비)
    N, C, H, W = input_data.shape
    
    # 출력 데이터의 높이와 너비 계산
    out_h = (H + 2*pad - filter_h) // stride + 1
    out_w = (W + 2*pad - filter_w) // stride + 1

    # 입력 데이터에 패딩 추가
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    # img는 입력 데이터에 주어진 pad 값을 기준으로 패딩된 4차원 배열입니다.
    # (N, C, H, W) 구조의 배열에 대해 패딩을 (0, 0), (0, 0), (pad, pad), (pad, pad)로 설정합니다.
    # 이는 이미지 수(N)와 채널 수(C)에는 패딩을 추가하지 않고, 높이(H)와 너비(W)에만 pad 폭만큼 상하좌우에 0으로 채웁니다.

    # 필터가 적용될 위치에 따라 필요한 데이터를 담을 배열 초기화
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    # 필터의 각 위치에 따라 슬라이싱하여 col 배열에 데이터 채우기
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    # col 배열을 (N, C, filter_h, filter_w, out_h, out_w) -> (N*out_h*out_w, C*filter_h*filter_w)로 변환
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col



def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """(im2col과 반대) 2차원 배열을 입력받아 다수의 이미지 묶음으로 변환한다.
    
    Parameters
    ----------
    col : 2차원 배열(입력 데이터)
    input_shape : 원래 이미지 데이터의 형상（예：(10, 1, 28, 28)）
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩
    
    Returns
    -------
    img : 변환된 이미지들
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]
