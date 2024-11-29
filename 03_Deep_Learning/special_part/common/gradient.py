# coding: utf-8
import numpy as np

def _numerical_gradient_1d(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 값 복원
        
    return grad


def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_1d(f, x)
        
        return grad


def numerical_gradient(f, x):
    h = 1e-4  # 작은 변화량 h 정의 (0.0001)
    grad = np.zeros_like(x)  # x와 같은 크기의 배열을 생성하여 미분값(기울기)을 저장

    # x 배열을 반복(iterate)하기 위한 nditer 객체 생성
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:  # 모든 요소에 대해 반복  
        idx = it.multi_index  # 현재 요소의 인덱스 가져오기
        tmp_val = x[idx]  # 현재 요소 값을 임시로 저장
        
        # f(x+h) 계산
        x[idx] = float(tmp_val) + h  # x의 현재 요소를 h만큼 증가
        fxh1 = f(x)  # f(x + h) 계산 x는 현재 x + h임
        
        # f(x-h) 계산
        x[idx] = tmp_val - h  # x의 현재 요소를 h만큼 감소
        fxh2 = f(x)  # f(x - h) 계산
        
        # 수치 미분 (중앙 차분) 계산하여 grad 배열에 저장
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        
        # x의 현재 요소 값을 원래 값으로 복원
        x[idx] = tmp_val
        it.iternext()  # 다음 요소로 이동
        
    return grad  # 모든 요소에 대한 기울기를 반환


